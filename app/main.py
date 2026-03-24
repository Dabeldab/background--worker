"""
Media worker: YouTube URL → captions or Whisper transcript for n8n theme extraction.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="media-worker", version="0.1.0")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
SUB_LANGS = os.environ.get("YTDLP_SUB_LANGS", "en.*,en-US.*")

_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model=%s device=%s compute=%s",
            WHISPER_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
    return _whisper_model


class YouTubeContextRequest(BaseModel):
    url: str = Field(..., description="Full YouTube watch or youtu.be URL")
    prefer_captions: bool = Field(default=True)


class YouTubeContextResponse(BaseModel):
    source: str  # captions | whisper
    text: str
    whisper_model: str | None = None
    note: str | None = None


def run_cmd(args: list[str], cwd: str | None = None) -> None:
    p = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout or f"command failed: {args}")


def vtt_to_plain(vtt_path: Path) -> str:
    raw = vtt_path.read_text(encoding="utf-8", errors="ignore")
    # Drop WEBVTT headers, timestamps, cue settings
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}", line) or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        # Strip simple HTML tags from auto-captions
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_subtitle_files(d: Path) -> list[Path]:
    return sorted(d.glob("*.vtt")) + sorted(d.glob("*.srt"))


@app.get("/health")
def health():
    return {"ok": True, "whisper_model": WHISPER_MODEL}


@app.post("/youtube-context", response_model=YouTubeContextResponse)
def youtube_context(body: YouTubeContextRequest):
    url = body.url.strip()
    if not url or "youtu" not in url.lower():
        raise HTTPException(status_code=400, detail="Invalid or missing YouTube URL")

    tmp = tempfile.mkdtemp(prefix="mw-")
    try:
        # --- 1) Try subtitles (manual + auto) ---
        if body.prefer_captions:
            sub_base = str(Path(tmp) / "sub")
            try:
                run_cmd(
                    [
                        "yt-dlp",
                        "--write-subs",
                        "--write-auto-subs",
                        "--sub-langs",
                        SUB_LANGS,
                        "--skip-download",
                        "-o",
                        sub_base + ".%(ext)s",
                        url,
                    ],
                    cwd=tmp,
                )
            except RuntimeError as e:
                logger.warning("yt-dlp subs failed (will try audio): %s", e)

            for p in find_subtitle_files(Path(tmp)):
                plain = vtt_to_plain(p) if p.suffix == ".vtt" else p.read_text(errors="ignore")
                if p.suffix == ".srt":
                    plain = re.sub(r"^\d+\s*$", "", plain, flags=re.MULTILINE)
                    plain = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", plain)
                    plain = re.sub(r"\s+", " ", plain).strip()
                if len(plain) >= 80:
                    return YouTubeContextResponse(
                        source="captions",
                        text=plain[:50000],
                        whisper_model=None,
                        note=f"parsed {p.name}",
                    )

        # --- 2) Audio → mono 16 kHz WAV → Whisper ---
        audio_tpl = str(Path(tmp) / "%(id)s.%(ext)s")
        run_cmd(
            [
                "yt-dlp",
                "-f",
                "bestaudio/best",
                "-x",
                "--audio-format",
                "wav",
                "--ppa",
                "ffmpeg:-ac 1 -ar 16000",
                "-o",
                audio_tpl,
                url,
            ],
            cwd=tmp,
        )
        wavs = sorted(Path(tmp).glob("*.wav"))
        if not wavs:
            raise HTTPException(
                status_code=502,
                detail="Could not download or extract audio",
            )
        wav_path = wavs[0]

        model = get_whisper()
        segments, _info = model.transcribe(
            str(wav_path),
            beam_size=5,
            vad_filter=True,
        )
        parts = [s.text.strip() for s in segments if s.text]
        text = " ".join(parts).strip()
        if not text:
            raise HTTPException(status_code=502, detail="Whisper returned empty transcript")

        return YouTubeContextResponse(
            source="whisper",
            text=text[:50000],
            whisper_model=WHISPER_MODEL,
            note=None,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/transcribe-wav")
async def transcribe_wav():
    """Optional: multipart upload for non-YouTube audio (extend later)."""
    raise HTTPException(status_code=501, detail="Use /youtube-context for now")
