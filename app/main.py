"""
Media worker: YouTube URL → captions or Whisper transcript for n8n theme extraction.
Audiogram: ElevenLabs audio → branded waveform MP4 for social sharing.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="media-worker", version="0.2.0")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
SUB_LANGS = os.environ.get("YTDLP_SUB_LANGS", "en.*,en-US.*")

_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
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
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}", line) or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
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

        audio_tpl = str(Path(tmp) / "%(id)s.%(ext)s")
        run_cmd(
            [
                "yt-dlp",
                "-f", "bestaudio/best",
                "-x", "--audio-format", "wav",
                "--ppa", "ffmpeg:-ac 1 -ar 16000",
                "-o", audio_tpl,
                url,
            ],
            cwd=tmp,
        )
        wavs = sorted(Path(tmp).glob("*.wav"))
        if not wavs:
            raise HTTPException(status_code=502, detail="Could not download or extract audio")
        wav_path = wavs[0]

        model = get_whisper()
        segments, _info = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
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


# ── Audiogram ──────────────────────────────────────────────────────────────────
# Receives ElevenLabs MP3 audio + episode metadata, returns a branded
# waveform MP4 ready to post to Instagram Reels / YouTube Shorts / TikTok.
# ──────────────────────────────────────────────────────────────────────────────

RENDER_DIR = Path(tempfile.gettempdir()) / "audiograms"
RENDER_DIR.mkdir(exist_ok=True)

_RENDER_JOB_DIR: ContextVar[Path | None] = ContextVar("render_job_dir", default=None)


def _write_render_debug_file(name: str, content: str) -> None:
    job_dir = _RENDER_JOB_DIR.get()
    if not job_dir:
        return
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / name).write_text(str(content), encoding="utf-8")
    except Exception as e:
        logger.warning("debug file write failed name=%s error=%s", name, e)


# Walk With Me brand palette
_BG      = "#0D1B2A"   # deep navy
_ACCENT  = "#E8A838"   # warm gold  (hex as int for FFmpeg drawtext: 0xE8A838)
_TEXT    = "#F5F0E8"   # warm white
_SUB     = "#8FA8BF"   # muted blue-grey
_WAVE_BG = "#111C28"   # slightly lighter navy for waveform band

# Card dimensions — 1080×1920 = 9:16 vertical (Reels/Shorts/TikTok)
# Change to 1080×1080 for square Instagram feed posts
_W, _H = 1080, 1920
_WAVE_TOP = 880    # y-offset where waveform band starts
_WAVE_H   = 480    # height of waveform band

# Font search order — first match wins; falls back to PIL default
_FONT_PATHS = [
    "/app/fonts/Merriweather-Bold.ttf",
    "/app/fonts/OpenSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
]


def _load_font(size: int):
    """Load first available TTF at the requested size, else PIL default."""
    from PIL import ImageFont
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


def _make_background(title: str, scripture: str, day_label: str, out: Path) -> None:
    """Generate the static branded card that FFmpeg will overlay the waveform on."""
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (_W, _H), _hex_to_rgb(_BG))
    draw = ImageDraw.Draw(img)

    # Top gold accent bar
    draw.rectangle([(0, 0), (_W, 14)], fill=_hex_to_rgb(_ACCENT))

    # Brand name
    brand_font = _load_font(56)
    draw.text((_W // 2, 110), "WALK WITH ME", font=brand_font,
              fill=_hex_to_rgb(_ACCENT), anchor="mm")

    # Day label  e.g. "Day 47 · Tuesday"
    day_font = _load_font(38)
    draw.text((_W // 2, 185), day_label or "", font=day_font,
              fill=_hex_to_rgb(_SUB), anchor="mm")

    # Rule
    draw.rectangle([(80, 228), (_W - 80, 232)], fill=_hex_to_rgb(_ACCENT))

    # Episode title — wrap at ~22 chars per line
    title_font = _load_font(72)
    lines = textwrap.wrap(title or "Daily Devotional", width=22)
    y = 330
    for line in lines:
        draw.text((_W // 2, y), line, font=title_font,
                  fill=_hex_to_rgb(_TEXT), anchor="mm")
        y += 96

    # Waveform zone (darker band — FFmpeg animates here)
    draw.rectangle([(0, _WAVE_TOP), (_W, _WAVE_TOP + _WAVE_H)],
                   fill=_hex_to_rgb(_WAVE_BG))

    # Scripture — below waveform band
    scrip_font = _load_font(44)
    scrip_lines = textwrap.wrap(scripture or "", width=30)
    y = _WAVE_TOP + _WAVE_H + 60
    for line in scrip_lines:
        draw.text((_W // 2, y), line, font=scrip_font,
                  fill=_hex_to_rgb(_ACCENT), anchor="mm")
        y += 62

    # Bottom gold accent bar
    draw.rectangle([(0, _H - 14), (_W, _H)], fill=_hex_to_rgb(_ACCENT))

    img.save(str(out), format="PNG")


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y"] + args
    cmd_text = " ".join(cmd)
    logger.info("FFmpeg: %s", cmd_text)
    _write_render_debug_file("last_ffmpeg_cmd.txt", cmd_text)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
    _write_render_debug_file("last_ffmpeg_stdout.log", result.stdout or "")
    _write_render_debug_file("last_ffmpeg_stderr.log", result.stderr or "")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-8000:]
        raise RuntimeError(f"FFmpeg failed (see last_ffmpeg_stderr.log):\n{tail}")
    return result


# Final encode: streaming/social loudness (~−14 LUFS) + true-peak limiter.
_WWM_AUDIO_MASTER = (
    "loudnorm=I=-14:TP=-1.0:LRA=11,"
    "alimiter=limit=0.98:attack=2:release=50"
)
# Light sharpen after scale/crop (reduces mush from aggressive scaling).
_WWM_VIDEO_SHARPEN = "unsharp=5:5:0.65:3:3:0.0"


def _prenorm_audio(src: Path, dst: Path) -> Path:
    """Normalize loudness in an isolated audio-only pass.

    Runs loudnorm + alimiter as a standalone subprocess so the two-pass
    loudnorm buffering never blocks a looping video input or -shortest.
    Returns dst.
    """
    _run_ffmpeg([
        "-i", str(src),
        "-af", _WWM_AUDIO_MASTER,
        "-c:a", "aac", "-b:a", "192k", "-vn",
        str(dst),
    ])
    return dst


@app.post("/render-audiogram")
async def render_audiogram(
    audio:     UploadFile = File(..., description="MP3/WAV from ElevenLabs"),
    title:     str = Form(default="Daily Devotional"),
    scripture: str = Form(default=""),
    day_label: str = Form(default=""),
    style:     str = Form(default="waveform"),   # "waveform" | "spectrum"
    quality:   str = Form(default="medium"),     # "fast" | "medium" | "hq"
):
    """
    Accepts a multipart POST with an audio file + episode metadata.
    Returns a branded 9:16 MP4 audiogram for social sharing.

    Fields:
      audio      — binary MP3 or WAV (required)
      title      — episode title text
      scripture  — scripture reference shown below waveform
      day_label  — e.g. "Day 47 · Tuesday"
      style      — "waveform" (default) or "spectrum"
      quality    — "fast" | "medium" (default) | "hq"
    """
    job_id  = uuid.uuid4().hex[:12]
    job_dir = RENDER_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_token = _RENDER_JOB_DIR.set(job_dir)

    audio_path = job_dir / "input.mp3"
    bg_path    = job_dir / "background.png"
    out_path   = job_dir / "audiogram.mp4"

    try:
        # 1. Save uploaded audio
        audio_bytes = await audio.read()
        audio_path.write_bytes(audio_bytes)
        logger.info("Audiogram job %s — audio %d bytes", job_id, len(audio_bytes))

        # 2. Generate branded background card
        _make_background(title, scripture, day_label, bg_path)

        # 3. Build FFmpeg filter graph
        preset_map = {"fast": "ultrafast", "medium": "fast", "hq": "slow"}
        preset     = preset_map.get(quality, "fast")

        if style == "spectrum":
            visualizer = (
                f"[a_vis]showspectrum="
                f"s={_W}x{_WAVE_H}:mode=combined:slide=scroll"
                f":color=rainbow:scale=cbrt[viz]"
            )
        else:
            # Gold single-line waveform, 30 fps
            visualizer = (
                f"[a_vis]showwaves="
                f"s={_W}x{_WAVE_H}:mode=cline"
                f":colors=0xE8A838|0xE8A838:rate=30[viz]"
            )

        # Pre-norm audio first so loudnorm never blocks inside filter_complex.
        # Split audio: one branch for viz, one direct to encoder (already normed).
        normed_path = _prenorm_audio(audio_path, job_dir / "normed.aac")
        filter_complex = (
            f"[0:a]asplit=2[a_vis][a_enc];"
            f"{visualizer};"
            f"[1:v][viz]overlay=0:{_WAVE_TOP}[out]"
        )

        _run_ffmpeg([
            "-i",      str(normed_path),
            "-loop",   "1",
            "-i",      str(bg_path),
            "-filter_complex", filter_complex,
            "-map",    "[out]",
            "-map",    "[a_enc]",
            "-c:v",    "libx264",
            "-preset", preset,
            "-crf",    "22",
            "-c:a",    "aac",
            "-b:a",    "192k",
            "-pix_fmt","yuv420p",
            "-shortest",
            str(out_path),
        ])

        logger.info("Audiogram job %s — render complete: %s", job_id, out_path)

        return FileResponse(
            str(out_path),
            media_type="video/mp4",
            filename=f"wwm_devotional_{job_id}.mp4",
            headers={"X-Job-Id": job_id},
        )

    except RuntimeError as e:
        logger.error("Audiogram job %s — FFmpeg error: %s", job_id, str(e)[:500])
        raise HTTPException(status_code=500, detail=f"FFmpeg render failed: {str(e)[:500]}")
    except Exception as e:
        logger.error("Audiogram job %s — unexpected error: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _RENDER_JOB_DIR.reset(job_token)
        # Clean up job folder after response is sent
        # (FileResponse streams before this runs, so the file is safe)
        def _cleanup():
            time.sleep(60)          # keep for 60 s in case of retries
            shutil.rmtree(job_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()


# ── Dynamic Multi-Format Renderer ─────────────────────────────────────────────
# LLM-directed video generation: audiogram | captioned | slideshow |
# captioned_slideshow | scripture_cards | full
# Pixabay background music mixed at low volume with graceful fade-out.
# ──────────────────────────────────────────────────────────────────────────────

import json
import urllib.request as _urllib_req


def _get_audio_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return float(stream.get("duration", 0))
    except Exception as e:
        logger.warning("ffprobe duration failed: %s", e)
    return 0.0


def _seconds_to_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a remote file to dest. Returns True on success."""
    try:
        req = _urllib_req.Request(url, headers={"User-Agent": "WalkWithMe-Worker/1.0"})
        with _urllib_req.urlopen(req, timeout=timeout) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:
        logger.warning("Download failed %s: %s", url, e)
        return False




def _seconds_to_ass_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    cs = int(round((s % 1) * 100))
    if cs >= 100:
        sec += 1
        cs = 0
    return f"{h:d}:{m:02d}:{sec:02d}.{cs:02d}"


def _normalize_caption_mode(mode: str, fallback: str = "phrase_safe") -> str:
    mode = str(mode or fallback).strip().lower()
    allowed = {"phrase_safe", "pop_word", "karaoke_dynamic", "landing_emphasis"}
    return mode if mode in allowed else fallback


def _caption_target_words(mode: str, max_words: int = 4) -> int:
    max_words = max(1, int(max_words or 4))
    if mode == "pop_word":
        return 1
    if mode == "landing_emphasis":
        return min(max_words, 2)
    if mode == "karaoke_dynamic":
        return min(max_words, 3)
    return min(max_words, 5)


def _caption_min_hold(mode: str, density_hint: str = "balanced") -> float:
    density_hint = str(density_hint or "balanced").strip().lower()
    base = {
        "pop_word": 0.18,
        "landing_emphasis": 0.24,
        "karaoke_dynamic": 0.22,
        "phrase_safe": 0.38,
    }.get(mode, 0.28)
    if density_hint == "slower":
        return base + 0.08
    if density_hint == "faster":
        return max(0.10, base - 0.05)
    return base


def _split_caption_lines(words: list[str], max_chars: int = 22, max_lines: int = 2) -> str:
    cleaned = [str(w or "").strip() for w in words if str(w or "").strip()]
    if not cleaned:
        return ""
    max_chars = max(8, int(max_chars or 22))
    max_lines = max(1, int(max_lines or 2))
    lines: list[str] = []
    current = []
    for word in cleaned:
        candidate = (" ".join(current + [word])).strip()
        if current and len(candidate) > max_chars and len(lines) < max_lines - 1:
            lines.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current).strip())
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max_chars].rstrip()
    return r"\N".join(lines)


def _extract_timed_words(segments) -> list[dict]:
    out: list[dict] = []
    for seg in segments:
        words = getattr(seg, "words", None) or []
        if words:
            for word in words:
                start = float(getattr(word, "start", getattr(seg, "start", 0.0)) or 0.0)
                end = float(getattr(word, "end", start) or start)
                token = str(getattr(word, "word", "") or "").strip()
                if token:
                    out.append({"start": start, "end": max(end, start + 0.02), "text": token})
            continue
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        tokens = [t for t in re.split(r"\s+", text) if t]
        if not tokens:
            continue
        seg_start = float(getattr(seg, "start", 0.0) or 0.0)
        seg_end = float(getattr(seg, "end", seg_start) or seg_start)
        seg_dur = max(seg_end - seg_start, 0.06 * len(tokens))
        step = seg_dur / max(len(tokens), 1)
        for idx, token in enumerate(tokens):
            start = seg_start + (idx * step)
            end = seg_start + ((idx + 1) * step)
            out.append({"start": start, "end": max(end, start + 0.02), "text": token})
    return out


def _build_caption_cues(words: list[dict], *, caption_mode: str = "phrase_safe", caption_hook_mode: str = "pop_word", caption_landing_mode: str = "landing_emphasis", max_words_per_caption: int = 4, max_chars_per_line: int = 22, max_lines: int = 2, density_hint: str = "balanced") -> list[dict]:
    if not words:
        return []
    total_words = len(words)
    hook_cutoff = max(6, int(total_words * 0.14))
    landing_cutoff = max(6, int(total_words * 0.16))
    cues: list[dict] = []
    i = 0
    while i < total_words:
        if i < hook_cutoff:
            mode = _normalize_caption_mode(caption_hook_mode, caption_mode)
        elif i >= max(0, total_words - landing_cutoff):
            mode = _normalize_caption_mode(caption_landing_mode, caption_mode)
        else:
            mode = _normalize_caption_mode(caption_mode)

        target_words = _caption_target_words(mode, max_words_per_caption)
        selected = []
        j = i
        while j < total_words:
            candidate = selected + [words[j]]
            candidate_text = _split_caption_lines([w["text"] for w in candidate], max_chars_per_line, max_lines)
            if selected and (len(candidate) > target_words or candidate_text.count(r"\N") + 1 > max_lines):
                break
            selected = candidate
            if len(selected) >= target_words:
                break
            j += 1
            if j >= total_words:
                break
        if not selected:
            selected = [words[i]]
        start = float(selected[0]["start"])
        end = float(selected[-1]["end"])
        end = max(end, start + _caption_min_hold(mode, density_hint))
        text = _split_caption_lines([w["text"] for w in selected], max_chars_per_line, max_lines)
        if text:
            cues.append({"start": start, "end": end, "text": text, "style": mode})
        i += len(selected)

    for idx in range(len(cues) - 1):
        next_start = cues[idx + 1]["start"]
        cues[idx]["end"] = min(cues[idx]["end"], max(cues[idx]["start"] + 0.06, next_start - 0.02))
    for cue in cues:
        cue["end"] = max(cue["end"], cue["start"] + 0.06)
    return cues


def _caption_margin_v(render_height: int = 1920, zone: str = "lower_third_safe", safe_margin: float = 0.12) -> int:
    render_height = max(int(render_height or 1920), 720)
    zone = str(zone or "lower_third_safe").strip().lower()
    safe_margin = max(0.05, min(float(safe_margin or 0.12), 0.25))
    if zone == "mid_low_safe":
        return max(int(render_height * (safe_margin + 0.14)), 340 if render_height >= 1600 else 220)
    if zone == "center_safe":
        return 0
    return max(int(render_height * (safe_margin + 0.09)), 260 if render_height >= 1600 else 170)


def _caption_ass_styles(render_width: int = 1080, render_height: int = 1920, zone: str = "lower_third_safe", safe_margin: float = 0.12) -> dict[str, str]:
    margin_lr = max(int(render_width * max(0.05, min(float(safe_margin or 0.12), 0.22))), 72)
    margin_v = _caption_margin_v(render_height, zone, safe_margin)
    align = "2" if str(zone or "").strip().lower() != "center_safe" else "5"
    styles = {
        "phrase_safe": f"Style: PhraseSafe,DejaVu Sans,{30 if render_height >= 1600 else 24},&H00FFFFFF,&H00FFFFFF,&H00000000,&H66000000,1,0,0,0,100,100,0,0,4,2.4,0,{align},{margin_lr},{margin_lr},{margin_v},1",
        "pop_word": f"Style: PopWord,DejaVu Sans,{44 if render_height >= 1600 else 34},&H00FFFFFF,&H00FFFFFF,&H00000000,&H66000000,1,0,0,0,100,100,0,0,4,2.8,0,{align},{margin_lr},{margin_lr},{margin_v},1",
        "karaoke_dynamic": f"Style: KaraokeDynamic,DejaVu Sans,{36 if render_height >= 1600 else 28},&H00FFFFFF,&H00FFFFFF,&H00000000,&H66000000,1,0,0,0,100,100,0,0,4,2.6,0,{align},{margin_lr},{margin_lr},{margin_v},1",
        "landing_emphasis": f"Style: LandingEmphasis,DejaVu Sans,{48 if render_height >= 1600 else 36},&H00FFFFFF,&H00FFFFFF,&H00000000,&H66000000,1,0,0,0,100,100,0,0,4,3.0,0,{align},{margin_lr},{margin_lr},{margin_v},1",
    }
    return styles


def _safe_ass_text(text: str) -> str:
    return str(text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _generate_srt(audio_path: Path, job_dir: Path, *, caption_mode: str = "phrase_safe", caption_hook_mode: str = "pop_word", caption_landing_mode: str = "landing_emphasis", max_words_per_caption: int = 4, max_chars_per_line: int = 22, max_lines: int = 2, portrait_safe_margin: float = 0.12, caption_vertical_zone: str = "lower_third_safe", dynamic_word_emphasis: bool = True, caption_animation_level: str = "medium", caption_density_hint: str = "balanced", render_width: int = 1080, render_height: int = 1920) -> Path | None:
    """Run Whisper on audio and write portrait-safe ASS subtitles for social video."""
    try:
        model = get_whisper()
        try:
            segments, _ = model.transcribe(
                str(audio_path), beam_size=5, vad_filter=True, word_timestamps=True,
            )
        except TypeError:
            segments, _ = model.transcribe(
                str(audio_path), beam_size=5, vad_filter=True,
            )

        timed_words = _extract_timed_words(segments)
        if not timed_words:
            return None

        if not dynamic_word_emphasis and caption_mode == "pop_word":
            caption_mode = "phrase_safe"

        cues = _build_caption_cues(
            timed_words,
            caption_mode=_normalize_caption_mode(caption_mode),
            caption_hook_mode=_normalize_caption_mode(caption_hook_mode, "pop_word"),
            caption_landing_mode=_normalize_caption_mode(caption_landing_mode, "landing_emphasis"),
            max_words_per_caption=max_words_per_caption,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
            density_hint=caption_density_hint,
        )
        if not cues:
            return None

        styles = _caption_ass_styles(render_width, render_height, caption_vertical_zone, portrait_safe_margin)
        ass_path = job_dir / "captions.ass"
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {render_width}",
            f"PlayResY: {render_height}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 2",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            styles["phrase_safe"],
            styles["pop_word"],
            styles["karaoke_dynamic"],
            styles["landing_emphasis"],
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        ]
        style_map = {
            "phrase_safe": "PhraseSafe",
            "pop_word": "PopWord",
            "karaoke_dynamic": "KaraokeDynamic",
            "landing_emphasis": "LandingEmphasis",
        }
        for cue in cues:
            lines.append(
                f"Dialogue: 0,{_seconds_to_ass_time(cue['start'])},{_seconds_to_ass_time(cue['end'])},{style_map.get(cue['style'], 'PhraseSafe')},,0,0,0,,{_safe_ass_text(cue['text'])}"
            )
        ass_path.write_text("\n".join(lines), encoding="utf-8")
        return ass_path
    except Exception as e:
        logger.warning("SRT generation failed: %s", e)
        return None


def _mix_audio(voice_path: Path, music_path: Path, out_path: Path,
               music_vol: float = 0.08, fade_secs: int = 5) -> Path:
    """
    Mix voice audio with background music at low volume.
    Music fades out over the last `fade_secs` seconds so it never cuts abruptly.
    Returns out_path.
    """
    duration   = _get_audio_duration(voice_path)
    fade_start = max(0.0, duration - fade_secs)

    _run_ffmpeg([
        "-i", str(voice_path),
        "-stream_loop", "-1",      # loop music if shorter than voice
        "-i", str(music_path),
        "-filter_complex",
        (
            f"[1:a]volume={music_vol},"
            f"afade=t=out:st={fade_start:.2f}:d={fade_secs}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[out]"
        ),
        "-map", "[out]",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ])
    return out_path


def _measure_or_estimate_duration(path: Path, estimated: float = 0.0) -> float:
    measured = _get_audio_duration(path)
    if measured > 0:
        return measured
    return max(float(estimated or 0.0), 0.0)


def _safe_turn_filename(turn_index: int, speaker_id: str, url: str) -> str:
    speaker_slug = re.sub(r"[^a-z0-9_-]+", "_", str(speaker_id or "speaker").lower()).strip("_") or "speaker"
    suffix = Path(url.split("?", 1)[0]).suffix or ".mp3"
    return f"turn_{turn_index + 1:03d}_{speaker_slug}{suffix}"


def _coerce_public_storage_url(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value.startswith("gs://"):
        return "https://storage.googleapis.com/" + value.replace("gs://", "", 1)
    return value


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _coerce_float(value, default: float) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _ambient_filter_label(seed: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", str(seed or "").lower()) or "amb"
    return f"{slug}{index}"


def _append_ambient_mix(
    filter_parts: list[str],
    dialogue_label: str,
    ambient_input_index: int,
    ambient_volume: float,
    fade_out_start: float,
    ducking: bool,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    makeup: float,
) -> str:
    filter_parts.append(
        f"[{ambient_input_index}:a]volume={ambient_volume},"
        f"afade=t=in:st=0:d=2,"
        f"afade=t=out:st={fade_out_start:.2f}:d=3[amb]"
    )
    if ducking:
        filter_parts.append(
            f"[amb][{dialogue_label}]sidechaincompress="
            f"threshold={threshold}:ratio={ratio}:attack={attack}:release={release}:makeup={makeup}"
            f"[ambduck]"
        )
        filter_parts.append(
            f"[{dialogue_label}][ambduck]amix=inputs=2:duration=longest:dropout_transition=0[amix]"
        )
    else:
        filter_parts.append(
            f"[{dialogue_label}][amb]amix=inputs=2:duration=longest:dropout_transition=0[amix]"
        )
    return "[amix]"


def _append_ambient_layers_mix(
    filter_parts: list[str],
    dialogue_label: str,
    current_mix_label: str,
    ambient_layers: list[dict],
    total_duration: float,
) -> str:
    mix_label = current_mix_label
    for idx, layer in enumerate(ambient_layers):
        input_index = int(layer["input_index"])
        volume = max(float(layer["volume"]), 0.0)
        ducking = bool(layer["ducking"])
        threshold = float(layer["duck_threshold"])
        ratio = float(layer["duck_ratio"])
        attack = float(layer["duck_attack"])
        release = float(layer["duck_release"])
        makeup = float(layer["duck_makeup"])
        fade_in = max(float(layer["fade_in"]), 0.0)
        fade_out = max(float(layer["fade_out"]), 0.0)
        layer_name = layer["name"]
        base_label = _ambient_filter_label(layer_name, idx)
        bed_label = f"{base_label}_bed"
        duck_label = f"{base_label}_duck"
        next_mix_label = f"{base_label}_mix"
        filter_chain = [f"volume={volume:.4f}"]
        if fade_in > 0:
            filter_chain.append(f"afade=t=in:st=0:d={fade_in:.2f}")
        if fade_out > 0:
            fade_out_start = max(0.0, total_duration - fade_out)
            filter_chain.append(f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}")
        filter_parts.append(
            f"[{input_index}:a]{','.join(filter_chain)}[{bed_label}]"
        )
        layer_mix_label = bed_label
        if ducking:
            filter_parts.append(
                f"[{bed_label}][{dialogue_label}]sidechaincompress="
                f"threshold={threshold}:ratio={ratio}:attack={attack}:release={release}:makeup={makeup}"
                f"[{duck_label}]"
            )
            layer_mix_label = duck_label
        filter_parts.append(
            f"[{mix_label}][{layer_mix_label}]amix=inputs=2:duration=longest:dropout_transition=0[{next_mix_label}]"
        )
        mix_label = next_mix_label
    return f"[{mix_label}]"


def _resolve_ambient_layer(raw_layer, *, default_volume: float, default_ducking: bool,
                           default_threshold: float, default_ratio: float,
                           default_attack: float, default_release: float,
                           default_makeup: float, default_name: str) -> dict | None:
    if raw_layer is None:
        return None
    if isinstance(raw_layer, str):
        payload = {"url": raw_layer}
    elif isinstance(raw_layer, dict):
        payload = raw_layer
    else:
        payload = raw_layer.model_dump()

    local_path = str(payload.get("local_path") or payload.get("path") or "").strip()
    url = _coerce_public_storage_url(
        payload.get("url")
        or payload.get("src")
        or payload.get("ambient_music_url")
        or payload.get("music_gcs_uri")
        or payload.get("gcs_uri")
        or payload.get("gcs_url")
    )
    if not url and not local_path:
        return None

    def pick(key: str, fallback):
        value = payload.get(key)
        return fallback if value is None else value

    return {
        "name": str(payload.get("name") or default_name),
        "url": url,
        "local_path": local_path,
        "volume": _coerce_float(pick("volume", default_volume), default_volume),
        "ducking": _coerce_bool(pick("ducking", default_ducking), default_ducking),
        "duck_threshold": _coerce_float(pick("duck_threshold", default_threshold), default_threshold),
        "duck_ratio": _coerce_float(pick("duck_ratio", default_ratio), default_ratio),
        "duck_attack": _coerce_float(pick("duck_attack", default_attack), default_attack),
        "duck_release": _coerce_float(pick("duck_release", default_release), default_release),
        "duck_makeup": _coerce_float(pick("duck_makeup", default_makeup), default_makeup),
        "fade_in": _coerce_float(pick("fade_in", 2.0), 2.0),
        "fade_out": _coerce_float(pick("fade_out", 3.0), 3.0),
    }


def _prepare_ambient_layers(job_dir: Path, body) -> list[dict]:
    layers: list[dict] = []

    primary_layer = _resolve_ambient_layer(
        {
            "name": "music",
            "url": body.ambient_music_url,
            "music_gcs_uri": body.music_gcs_uri,
            "local_path": body.music_local_path,
            "volume": body.ambient_volume,
            "ducking": body.ambient_ducking,
            "duck_threshold": body.ambient_duck_threshold,
            "duck_ratio": body.ambient_duck_ratio,
            "duck_attack": body.ambient_duck_attack,
            "duck_release": body.ambient_duck_release,
            "duck_makeup": body.ambient_duck_makeup,
            "fade_out": body.music_fade_out_seconds,
        },
        default_volume=body.ambient_volume,
        default_ducking=body.ambient_ducking,
        default_threshold=body.ambient_duck_threshold,
        default_ratio=body.ambient_duck_ratio,
        default_attack=body.ambient_duck_attack,
        default_release=body.ambient_duck_release,
        default_makeup=body.ambient_duck_makeup,
        default_name="music",
    )
    if primary_layer:
        layers.append(primary_layer)

    for idx, raw_layer in enumerate(body.ambient_fx_layers):
        layer = _resolve_ambient_layer(
            raw_layer,
            default_volume=0.025,
            default_ducking=False,
            default_threshold=body.ambient_duck_threshold,
            default_ratio=body.ambient_duck_ratio,
            default_attack=body.ambient_duck_attack,
            default_release=body.ambient_duck_release,
            default_makeup=body.ambient_duck_makeup,
            default_name=f"fx_{idx + 1}",
        )
        if layer:
            layers.append(layer)

    prepared_layers: list[dict] = []
    for idx, layer in enumerate(layers):
        source_hint = layer.get("local_path") or layer.get("url") or ""
        ext = Path(str(source_hint).split("?", 1)[0]).suffix or ".mp3"
        slug = re.sub(r"[^a-z0-9_-]+", "_", str(layer["name"]).lower()).strip("_") or f"ambient_{idx + 1}"
        path = job_dir / f"ambient_{idx + 1:02d}_{slug}{ext}"
        if not _stage_ambient_layer_asset(layer, path):
            logger.warning("ambient layer fetch failed name=%s src=%s", layer["name"], source_hint)
            continue
        if not path.exists() or path.stat().st_size == 0:
            logger.warning("ambient layer empty name=%s url=%s", layer["name"], layer["url"])
            continue
        prepared_layers.append({
            **layer,
            "path": path,
        })
    return prepared_layers


def _srt_force_style(style: str = "bottom") -> str:
    """Return an ASS/SRT force_style string for simple subtitle rendering."""
    base = (
        "FontName=DejaVu Sans,FontSize=28,Bold=1,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,BorderStyle=4,Outline=2,Shadow=0,"
        "MarginV=60,Alignment=2"
    )
    if style == "center":
        return base.replace("Alignment=2", "Alignment=5").replace("MarginV=60", "MarginV=0")
    if style == "top":
        return base.replace("Alignment=2", "Alignment=8").replace("MarginV=60", "MarginV=80")
    return base


def _subtitle_filter_clause(sub_path: Path, force_style: str = "") -> str:
    sub_escaped = str(sub_path).replace("\\", "/").replace(":", "\\:")
    if sub_path.suffix.lower() == ".ass":
        return f"subtitles='{sub_escaped}'"
    if force_style:
        return f"subtitles='{sub_escaped}':force_style='{force_style}'"
    return f"subtitles='{sub_escaped}'"


# ── Format renderers ───────────────────────────────────────────────────────────

def _fmt_audiogram(audio: Path, bg: Path, out: Path,
                   style: str = "waveform", preset: str = "fast") -> None:
    """Waveform/spectrum audiogram — no captions."""
    normed = _prenorm_audio(audio, audio.parent / (audio.stem + "_normed.aac"))
    if style == "spectrum":
        viz = (f"[a_vis]showspectrum=s={_W}x{_WAVE_H}:mode=combined"
               f":slide=scroll:color=rainbow:scale=cbrt[viz]")
    else:
        viz = (f"[a_vis]showwaves=s={_W}x{_WAVE_H}:mode=cline"
               f":colors=0xE8A838|0xE8A838:rate=30[viz]")
    fc = (
        f"[0:a]asplit=2[a_vis][a_enc];{viz};"
        f"[1:v][viz]overlay=0:{_WAVE_TOP}[out]"
    )
    _run_ffmpeg([
        "-i", str(normed), "-loop", "1", "-i", str(bg),
        "-filter_complex", fc,
        "-map", "[out]", "-map", "[a_enc]",
        "-c:v", "libx264", "-preset", preset, "-crf", "22",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest",
        str(out),
    ])


def _fmt_captioned(audio: Path, bg: Path, srt: Path, out: Path,
                   caption_style: str = "bottom", preset: str = "fast") -> None:
    """Waveform audiogram with burned-in captions."""
    normed = _prenorm_audio(audio, audio.parent / (audio.stem + "_normed.aac"))
    force_style = _srt_force_style(caption_style)
    srt_escaped = str(srt).replace("\\", "/").replace(":", "\\:")
    viz = (f"[a_vis]showwaves=s={_W}x{_WAVE_H}:mode=cline"
           f":colors=0xE8A838|0xE8A838:rate=30[viz]")
    fc = (
        f"[0:a]asplit=2[a_vis][a_enc];{viz};"
        f"[1:v][viz]overlay=0:{_WAVE_TOP}[waved];"
        f"[waved]subtitles='{srt_escaped}':force_style='{force_style}'[out]"
    )
    _run_ffmpeg([
        "-i", str(normed), "-loop", "1", "-i", str(bg),
        "-filter_complex", fc,
        "-map", "[out]", "-map", "[a_enc]",
        "-c:v", "libx264", "-preset", preset, "-crf", "22",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest",
        str(out),
    ])


def _fmt_slideshow(audio: Path, image_paths: list[Path], out: Path,
                   srt: Path | None = None, caption_style: str = "bottom",
                   preset: str = "fast") -> None:
    """
    Ken Burns slideshow: each image slowly zooms, images crossfade.
    Captions optionally burned in.
    Images are scaled to 1080 wide, placed on the branded navy background,
    with a blurred version filling the full 1920 height (portrait safe).
    """
    if not image_paths:
        raise ValueError("slideshow requires at least one image")

    normed     = _prenorm_audio(audio, audio.parent / (audio.stem + "_normed.aac"))
    duration   = _get_audio_duration(normed)
    n          = len(image_paths)
    clip_dur   = max(3.0, duration / n)   # minimum 3 s per image
    fps        = 25
    frame_cnt  = int(clip_dur * fps)

    # Build inputs list: normed audio first, then images
    inputs = ["-i", str(normed)]
    for p in image_paths:
        inputs += ["-loop", "1", "-t", str(clip_dur + 0.5), "-i", str(p)]

    filter_parts = []
    slide_labels = []

    for i, _ in enumerate(image_paths):
        idx = i + 1   # FFmpeg input index (0 = audio)
        label = f"slide{i}"

        # Zoom direction alternates to keep things dynamic
        zoom_expr = "min(zoom+0.0006,1.07)"
        x_expr    = "iw/2-(iw/zoom/2)" if i % 2 == 0 else "iw/2-(iw/zoom/2)+50"
        y_expr    = "ih/2-(ih/zoom/2)" if i % 2 == 0 else "ih/2-(ih/zoom/2)+30"

        filter_parts.append(
            f"[{idx}:v]"
            f"scale={_W}:-1:force_original_aspect_ratio=increase,"
            f"crop={_W}:{_H},{_WWM_VIDEO_SHARPEN},"
            f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={frame_cnt}:s={_W}x{_H}:fps={fps}"
            f"[{label}]"
        )
        slide_labels.append(f"[{label}]")

    # Crossfade between slides (0.5 s dissolve)
    if n == 1:
        filter_parts.append(f"[slide0]copy[slideshow]")
    else:
        prev = "slide0"
        for i in range(1, n):
            out_lbl = f"xf{i}" if i < n - 1 else "slideshow"
            offset  = clip_dur * i - 0.5
            filter_parts.append(
                f"[{prev}][slide{i}]xfade=transition=dissolve"
                f":duration=0.5:offset={offset:.2f}[{out_lbl}]"
            )
            prev = f"xf{i}"

    # Optionally burn captions
    if srt:
        fs   = _srt_force_style(caption_style)
        srt_e = str(srt).replace("\\", "/").replace(":", "\\:")
        filter_parts.append(
            f"[slideshow]subtitles='{srt_e}':force_style='{fs}'[out]"
        )
        video_pre_fade = "[out]"
    else:
        video_pre_fade = "[slideshow]"

    fade_st = max(0.0, duration - 0.5)
    fade_chain = (
        f"fade=t=in:st=0:d=0.4,fade=t=out:st={fade_st:.2f}:d=0.45"
    )
    filter_complex = (
        ";".join(filter_parts)
        + f";{video_pre_fade}{fade_chain}[vfinal]"
    )
    _run_ffmpeg(
        inputs + [
            "-filter_complex", filter_complex,
            "-map", "[vfinal]", "-map", "0:a",
            "-c:v", "libx264", "-preset", preset, "-crf", "22",
            "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest",
            str(out),
        ]
    )


def _fmt_scripture_cards(audio: Path, bg: Path, scripture: str,
                         title: str, out: Path, preset: str = "fast") -> None:
    """
    Animated text: episode title fades in at 0s, scripture fades in at 4s,
    both displayed over the branded background with the waveform.
    Good for devotional / meditative episodes.
    """
    normed = _prenorm_audio(audio, audio.parent / (audio.stem + "_normed.aac"))
    duration = _get_audio_duration(normed)

    # Escape special chars for FFmpeg drawtext
    def esc(t: str) -> str:
            return str(t).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    title_safe  = esc(title[:60])
    scrip_safe  = esc(scripture[:120])

    viz = (f"[a_vis]showwaves=s={_W}x{_WAVE_H}:mode=cline"
           f":colors=0xE8A838|0xE8A838:rate=30[viz]")

    drawtext = (
        # Episode title — appears from t=0, fades out at t=3
        f"drawtext=text='{title_safe}':fontsize=64:fontcolor=0xF5F0E8:"
        f"x=(w-text_w)/2:y=300:alpha='if(lt(t,0.5),t/0.5,if(lt(t,3),1,max(0,1-(t-3)/0.5)))',"
        # Scripture — fades in at t=4, stays for the rest
        f"drawtext=text='{scrip_safe}':fontsize=40:fontcolor=0xE8A838:"
        f"x=(w-text_w)/2:y=1500:alpha='if(lt(t,4),0,if(lt(t,4.8),(t-4)/0.8,1))'"
    )

    fc = (
        f"[0:a]asplit=2[a_vis][a_enc];{viz};"
        f"[1:v][viz]overlay=0:{_WAVE_TOP}[waved];[waved]{drawtext}[out]"
    )
    _run_ffmpeg([
        "-i", str(normed), "-loop", "1", "-i", str(bg),
        "-filter_complex", fc,
        "-map", "[out]", "-map", "[a_enc]",
        "-c:v", "libx264", "-preset", preset, "-crf", "22",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest",
        str(out),
    ])


def _fmt_full(audio: Path, bg: Path, image_paths: list[Path],
              srt: Path | None, scripture: str, title: str,
              out: Path, caption_style: str = "bottom",
              preset: str = "fast") -> None:
    """
    Full produced short: branded intro card (3s) → Ken Burns slideshow
    with waveform overlay at bottom + burned captions.
    Falls back to captioned audiogram if no images are provided.
    """
    if not image_paths:
        if srt:
            _fmt_captioned(audio, bg, srt, out, caption_style, preset)
        else:
            _fmt_audiogram(audio, bg, out, "waveform", preset)
        return

    # Split audio: first 3 s for intro card, then rest for slideshow
    duration = _get_audio_duration(audio)
    intro_dur = min(3.0, duration * 0.1)

    intro_out = out.parent / "intro.mp4"
    main_out  = out.parent / "main_slides.mp4"

    # Intro: scripture card with title
    _fmt_scripture_cards(audio, bg, scripture, title, intro_out, preset)

    # Main: slideshow with captions
    _fmt_slideshow(audio, image_paths, main_out, srt, caption_style, preset)

    # Concat intro + main
    list_file = out.parent / "concat.txt"
    list_file.write_text(
        f"file '{intro_out.name}'\nfile '{main_out.name}'\n",
        encoding="utf-8",
    )
    _run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out),
    ])


# ── /render-dynamic endpoint ───────────────────────────────────────────────────

@app.post("/render-dynamic")
async def render_dynamic(
    audio:             UploadFile = File(default=None, description="MP3/WAV binary — use this OR audio_url"),
    audio_url:         str  = Form(default=""),         # GCS/public URL alternative to binary upload
    video_format:      str  = Form(default="captioned"),
    title:             str  = Form(default="Daily Devotional"),
    scripture:         str  = Form(default=""),
    day_label:         str  = Form(default=""),
    image_urls:        str  = Form(default="[]"),   # JSON array of strings
    pixabay_music_url: str  = Form(default=""),
    music_volume:      float = Form(default=0.08),
    caption_style:     str  = Form(default="bottom"),  # bottom | center | top
    quality:           str  = Form(default="medium"),  # fast | medium | hq
    viz_style:         str  = Form(default="waveform"), # waveform | spectrum
):
    """
    LLM-directed multi-format video renderer.

    video_format options:
      audiogram           — waveform only (fastest)
      captioned           — waveform + auto-captions
      slideshow           — Ken Burns photo slideshow (no captions)
      captioned_slideshow — Ken Burns + captions
      scripture_cards     — animated title + scripture overlay on waveform
      full                — branded intro card + slideshow + captions

    Pixabay:
      image_urls        — JSON array of image URLs to download for slideshow formats
      pixabay_music_url — direct audio file URL; mixed at music_volume under voice
    """
    job_id  = uuid.uuid4().hex[:12]
    job_dir = RENDER_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_token = _RENDER_JOB_DIR.set(job_dir)
    logger.info("render-dynamic job=%s format=%s", job_id, video_format)

    preset_map = {"fast": "ultrafast", "medium": "fast", "hq": "slow"}
    preset = preset_map.get(quality, "fast")

    try:
        # ── 1. Obtain voice audio (binary upload OR remote URL) ───────────────
        raw_audio = job_dir / "voice_raw.mp3"
        if audio is not None and audio.filename:
            raw_audio.write_bytes(await audio.read())
        elif audio_url.strip():
            if not _download_file(audio_url.strip(), raw_audio, timeout=60):
                raise HTTPException(status_code=400,
                                    detail=f"Could not download audio from: {audio_url}")
        else:
            raise HTTPException(status_code=400,
                                detail="Provide either 'audio' (binary) or 'audio_url' (GCS/public URL)")

        # ── 2. Generate branded background card ───────────────────────────────
        bg_path = job_dir / "background.png"
        _make_background(title, scripture, day_label, bg_path)

        # ── 3. Download Pixabay background music and mix if provided ──────────
        voice_audio = raw_audio
        if pixabay_music_url.strip():
            music_path = job_dir / "bg_music.mp3"
            if _download_file(pixabay_music_url.strip(), music_path):
                mixed_path = job_dir / "voice_mixed.mp3"
                try:
                    _mix_audio(raw_audio, music_path, mixed_path,
                               music_vol=music_volume)
                    voice_audio = mixed_path
                    logger.info("job=%s music mixed at vol=%.2f", job_id, music_volume)
                except Exception as e:
                    logger.warning("job=%s music mix failed, using dry audio: %s", job_id, e)

        # ── 4. Download Pixabay images for slideshow formats ──────────────────
        image_paths: list[Path] = []
        needs_images = video_format in ("slideshow", "captioned_slideshow", "full")
        if needs_images:
            try:
                urls: list[str] = json.loads(image_urls)
            except Exception:
                urls = []
            for idx, url in enumerate(urls[:6]):   # cap at 6 images
                dest = job_dir / f"img_{idx:02d}.jpg"
                if _download_file(url, dest):
                    image_paths.append(dest)
            logger.info("job=%s downloaded %d images", job_id, len(image_paths))

        # ── 5. Generate SRT captions if needed ───────────────────────────────
        srt_path: Path | None = None
        needs_captions = video_format in ("captioned", "captioned_slideshow", "full")
        if needs_captions:
            logger.info("job=%s generating captions via Whisper", job_id)
            srt_path = _generate_srt(raw_audio, job_dir)
            if not srt_path:
                logger.warning("job=%s Whisper returned no segments, captions skipped", job_id)

        # ── 6. Render chosen format ───────────────────────────────────────────
        out_path = job_dir / "output.mp4"

        if video_format == "audiogram":
            _fmt_audiogram(voice_audio, bg_path, out_path, viz_style, preset)

        elif video_format == "captioned":
            if srt_path:
                _fmt_captioned(voice_audio, bg_path, srt_path, out_path, caption_style, preset)
            else:
                _fmt_audiogram(voice_audio, bg_path, out_path, viz_style, preset)

        elif video_format == "slideshow":
            if image_paths:
                _fmt_slideshow(voice_audio, image_paths, out_path,
                               srt=None, caption_style=caption_style, preset=preset)
            else:
                _fmt_audiogram(voice_audio, bg_path, out_path, viz_style, preset)

        elif video_format == "captioned_slideshow":
            if image_paths:
                _fmt_slideshow(voice_audio, image_paths, out_path,
                               srt=srt_path, caption_style=caption_style, preset=preset)
            else:
                _fmt_captioned(voice_audio, bg_path, srt_path or job_dir / "empty.srt",
                               out_path, caption_style, preset) if srt_path \
                    else _fmt_audiogram(voice_audio, bg_path, out_path, viz_style, preset)

        elif video_format == "scripture_cards":
            _fmt_scripture_cards(voice_audio, bg_path, scripture, title, out_path, preset)

        elif video_format == "full":
            _fmt_full(voice_audio, bg_path, image_paths, srt_path,
                      scripture, title, out_path, caption_style, preset)

        else:
            raise HTTPException(status_code=400,
                                detail=f"Unknown video_format: '{video_format}'")

        logger.info("job=%s render complete format=%s size=%d bytes",
                    job_id, video_format, out_path.stat().st_size)

        return FileResponse(
            str(out_path),
            media_type="video/mp4",
            filename=f"wwm_{video_format}_{job_id}.mp4",
            headers={
                "X-Job-Id":      job_id,
                "X-Video-Format": video_format,
                "X-Has-Captions": str(srt_path is not None).lower(),
                "X-Image-Count":  str(len(image_paths)),
            },
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("job=%s FFmpeg error: %s", job_id, str(e)[:500])
        raise HTTPException(status_code=500,
                            detail=f"Render failed [{video_format}]: {str(e)[:400]}")
    except Exception as e:
        logger.error("job=%s unexpected: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _RENDER_JOB_DIR.reset(job_token)
        def _cleanup():
            time.sleep(90)
            shutil.rmtree(job_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()


# ── /render-podcast — Shotstack replacement (podcast mode) ────────────────────
# Replicates: bg loop video + host audio + reflection audio + ambient music bed
# No API costs, no polling, 15-30s render on Hetzner vs 60-120s Shotstack wait
# ──────────────────────────────────────────────────────────────────────────────

class PodcastTurn(BaseModel):
    turn_index: int = Field(0, description="0-based order in the conversation")
    speaker_id: str = Field("SPEAKER_1", description="Speaker id for this turn")
    url: str = Field("", description="Public URL for the rendered turn audio")
    gcs_uri: str = Field("", description="Optional gs:// URI for the rendered turn audio")
    object_path: str = Field("", description="Optional object path for the rendered turn audio")
    local_path: str = Field("", description="Optional local filesystem path for the rendered turn audio")
    text: str = Field("", description="Turn text, used only for logging/debug")
    voice_energy: str = Field("anchor", description="Energy label from the voice writer")
    estimated_duration: float = Field(0.0, description="n8n-side duration estimate in seconds")
    gap_after: float = Field(0.06, description="Pause after this turn in seconds")
    sequence_scope: str = Field("full", description="full | short")


class AmbientFxLayer(BaseModel):
    name: str = Field("", description="Optional label for logs/debugging")
    url: str = Field("", description="Public URL for this ambient layer")
    src: str = Field("", description="Alternate public URL field")
    ambient_music_url: str = Field("", description="Alternate ambient URL field")
    music_gcs_uri: str = Field("", description="Alternate gs:// field")
    gcs_uri: str = Field("", description="Alternate gs:// field")
    gcs_url: str = Field("", description="Alternate public URL field")
    local_path: str = Field("", description="Optional local filesystem path")
    volume: float | None = Field(None, description="Layer volume 0.0-1.0")
    ducking: bool | None = Field(None, description="Whether to duck this layer under dialogue")
    duck_threshold: float | None = Field(None, description="Optional per-layer duck threshold")
    duck_ratio: float | None = Field(None, description="Optional per-layer duck ratio")
    duck_attack: float | None = Field(None, description="Optional per-layer duck attack")
    duck_release: float | None = Field(None, description="Optional per-layer duck release")
    duck_makeup: float | None = Field(None, description="Optional per-layer duck makeup gain")
    fade_in: float | None = Field(None, description="Fade-in duration in seconds")
    fade_out: float | None = Field(None, description="Fade-out duration in seconds")


class PodcastRenderRequest(BaseModel):
    bg_loop_url:        str   = Field("",  description="Public URL of background loop video")
    bg_loop_gcs_uri:    str   = Field("",  description="Optional gs:// background loop fallback")
    bg_object_path:     str   = Field("",  description="Optional bucket object path for the background loop")
    background_local_path: str = Field("", description="Optional local filesystem path to the background loop video")
    background_local_paths: list[str] = Field(
        default_factory=list,
        description="Optional list of local background videos to rotate through during the episode",
    )
    media_sequence: list[dict] = Field(
        default_factory=list,
        description="Optional ordered media sequence for local-first backgrounds",
    )
    outro_image_local_path: str = Field("", description="Optional local filesystem path to the branded outro image")
    outro_duration: float = Field(4.0, description="How long the outro card should hold, in seconds")
    outro_fade_duration: float = Field(1.2, description="Crossfade duration into the outro card, in seconds")
    host_audio_url:     str   = Field("",   description="GCS URL of host MP3 (empty when speaker_audio used)")
    host_audio_gcs_uri: str   = Field("",   description="Optional gs:// host audio fallback")
    host_audio_local_path: str = Field("", description="Optional local filesystem path for host audio")
    reflection_audio_url: str = Field("",   description="GCS URL of reflection MP3 (optional)")
    reflection_audio_gcs_uri: str = Field("", description="Optional gs:// reflection audio fallback")
    reflection_audio_local_path: str = Field("", description="Optional local filesystem path for reflection audio")
    # Multi-speaker fields sent by n8n Cast TTS node
    speaker_audio:      dict  = Field(default_factory=dict, description="SPEAKER_1..N dict from Cast TTS")
    full_turns:         list[PodcastTurn] = Field(default_factory=list, description="Full episode turn clips in playback order")
    short_turns:        list[PodcastTurn] = Field(default_factory=list, description="Short-cut turn clips in playback order")
    full_sequence:      list[dict] = Field(default_factory=list, description="Raw full combined_sequence from the writer")
    short_sequence:     list[dict] = Field(default_factory=list, description="Raw short combined_sequence from the writer")
    render_style:       str   = Field("speaker_fallback", description="interleaved_dialogue | speaker_fallback")
    turn_scope:         str   = Field("full", description="Prefer full or short turns when both are present")
    cast_size:          int   = Field(0,    description="Number of speakers (0 = derive from speaker_audio)")
    local_asset_preferred: bool = Field(False, description="Prefer local filesystem assets when available")
    run_id:             str   = Field("",   description="Episode run_id from n8n")
    media_temp_root:    str   = Field(
        "/mnt/HC_Volume_105210642/temp_runs",
        description="Root temp directory for run-scoped staging (paths derived for n8n)",
    )
    run_root:           str   = Field("",   description="Derived run root directory")
    audio_root:         str   = Field("",   description="Derived audio root directory")
    full_audio_root:    str   = Field("",   description="Derived full-audio directory")
    short_audio_root:   str   = Field("",   description="Derived short-audio directory")
    video_root:         str   = Field("",   description="Derived video output directory")
    clips_root:         str   = Field("",   description="Derived clips directory")
    episode_title:      str   = Field("",   description="Alias for title (n8n sends episode_title)")
    ambient_music_url:  str   = Field("",  description="GCS URL or Pixabay URL of ambient music bed")
    music_gcs_uri:      str   = Field("",  description="Legacy gs:// music URI fallback")
    music_local_path:   str   = Field("",  description="Optional local filesystem path to ambient music")
    ambient_source_mode: str  = Field("bucket", description="bucket | local | remote")
    ambient_scene_hint: str   = Field("", description="Optional scene hint for ambient search/selection")
    ambient_search_terms: list[str] = Field(default_factory=list, description="Optional ambient search terms")
    ambient_fx_layers:  list[AmbientFxLayer | str] = Field(default_factory=list, description="Optional extra ambient FX layers such as birds or water")
    host_duration:      float = Field(0,   description="Host audio duration in seconds (0 = auto-detect)")
    reflection_duration: float = Field(0,  description="Reflection audio duration in seconds (0 = auto-detect)")
    gap_seconds:        float = Field(0.5, description="Silence gap between host and reflection")
    ambient_volume:     float = Field(0.08,description="Ambient music volume 0.0–1.0")
    ambient_ducking:    bool  = Field(True, description="Lower ambient music automatically while dialogue is active")
    ambient_duck_threshold: float = Field(0.035, description="Sidechain threshold for ducking")
    ambient_duck_ratio: float = Field(8.0, description="Compression ratio used for ducking")
    ambient_duck_attack: float = Field(20.0, description="Attack time in ms for ducking")
    ambient_duck_release: float = Field(280.0, description="Release time in ms for ducking")
    ambient_duck_makeup: float = Field(1.0, description="Makeup gain after ducking compression")
    music_fade_out_seconds: float = Field(
        2.5,
        description="Fade-out duration applied to the primary music bed at the end",
    )
    aspect_ratio:       str   = Field("9:16", description="'9:16' (Reels/Shorts) or '16:9' (YouTube)")
    title:              str   = Field("",  description="Episode title for drawtext overlay")
    scripture:          str   = Field("",  description="Scripture reference for drawtext overlay")
    quality:            str   = Field("medium", description="fast | medium | hq")
    show_title:         bool  = Field(True, description="Burn episode title into the video")
    show_subtitles:     bool  = Field(True, description="Auto-generate and burn subtitles")
    audio_visual_style: str   = Field("waveform", description="waveform | spectrum | none")
    caption_style:      str   = Field("bottom", description="bottom | center | top")
    single_speaker_mode: bool = Field(False, description="When true, never synthesize or stitch a reflection track")
    caption_mode:       str   = Field("phrase_safe", description="phrase_safe | pop_word | karaoke_dynamic | landing_emphasis")
    caption_hook_mode:  str   = Field("pop_word", description="Caption mode for the opening hook")
    caption_landing_mode: str = Field("landing_emphasis", description="Caption mode for the close or landing line")
    max_words_per_caption: int = Field(4, description="Hard cap for words per caption chunk")
    max_chars_per_line: int = Field(22, description="Hard cap for characters per caption line in portrait")
    max_lines:          int   = Field(2, description="Maximum visible caption lines")
    portrait_safe_margin: float = Field(0.12, description="Side safe margin for portrait subtitles")
    caption_vertical_zone: str = Field("lower_third_safe", description="lower_third_safe | mid_low_safe | center_safe")
    dynamic_word_emphasis: bool = Field(True, description="Prefer denser, punchier word timing when available")
    caption_animation_level: str = Field("medium", description="low | medium | high")
    caption_density_hint: str = Field("balanced", description="slower | balanced | faster")


def _stage_local_or_remote_asset(source_url: str, dest_path: Path, *, local_path: str = "") -> bool:
    local = str(local_path or "").strip()
    if local:
        try:
            src = Path(local)
            if src.exists() and src.is_file():
                shutil.copyfile(src, dest_path)
                return dest_path.exists() and dest_path.stat().st_size > 0
        except Exception as e:
            logger.warning("local asset copy failed path=%s error=%s", local, e)
    source_url = str(source_url or "").strip()
    if not source_url:
        return False
    return _download_file(source_url, dest_path)


def _stage_ambient_layer_asset(layer: dict, dest_path: Path) -> bool:
    local = str(layer.get("local_path") or "").strip()
    if local:
        try:
            src = Path(local)
            if src.exists() and src.is_file():
                shutil.copyfile(src, dest_path)
                return dest_path.exists() and dest_path.stat().st_size > 0
        except Exception as e:
            logger.warning("ambient local copy failed path=%s error=%s", local, e)
    return _download_file(layer.get("url", ""), dest_path)


def _stage_any_asset(*, dest_path: Path, local_path: str = "", gcs_uri: str = "", object_path: str = "", url: str = "") -> bool:
    local = str(local_path or "").strip()
    if local:
        try:
            src = Path(local)
            if src.exists() and src.is_file():
                shutil.copyfile(src, dest_path)
                return dest_path.exists() and dest_path.stat().st_size > 0
        except Exception as e:
            logger.warning("local asset copy failed path=%s error=%s", local, e)
    remote = str(url or "").strip()
    if not remote:
        remote = _coerce_public_storage_url(gcs_uri or object_path)
    if not remote:
        return False
    return _download_file(remote, dest_path)


def _speaker_scope_value(entry: dict, scope: str, key: str) -> str:
    return str(
        entry.get(f"{scope}_{key}")
        or entry.get(key)
        or ""
    ).strip()


def _has_speaker_scope_source(entry: dict, scope: str = "full") -> bool:
    return bool(
        _speaker_scope_value(entry, scope, "local_path")
        or _speaker_scope_value(entry, scope, "gcs_uri")
        or _speaker_scope_value(entry, scope, "object_path")
        or _speaker_scope_value(entry, scope, "url")
    )


def _build_background_sequence(
    local_paths: list[str], total: float, job_dir: Path, W: int, H: int, preset: str
) -> Path | None:
    """Concatenate multiple local background clips into one loopable segment for the episode."""
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for raw in local_paths or []:
        p = str(raw or "").strip()
        if not p or p in seen:
            continue
        path = Path(p)
        if path.exists() and path.is_file():
            unique_paths.append(path)
            seen.add(p)
    if len(unique_paths) <= 1:
        return None

    seg_dur = max(float(total or 0.0) / max(len(unique_paths), 1), 3.0)
    inputs: list[str] = []
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for idx, src in enumerate(unique_paths):
        staged = job_dir / f"bg_src_{idx}{src.suffix.lower() or '.mp4'}"
        try:
            shutil.copyfile(src, staged)
        except OSError:
            staged = src
        inputs += ["-stream_loop", "-1", "-t", f"{seg_dur:.2f}", "-i", str(staged)]
        filter_parts.append(
            f"[{idx}:v]fps=30,scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},{_WWM_VIDEO_SHARPEN},setpts=PTS-STARTPTS,trim=duration={seg_dur:.2f}[v{idx}]"
        )
        concat_inputs.append(f"[v{idx}]")

    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={len(unique_paths)}:v=1:a=0[bgout]"
    )
    out_path = job_dir / "bg_sequence.mp4"
    _run_ffmpeg(
        inputs
        + [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[bgout]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{total:.2f}",
            str(out_path),
        ]
    )
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


@app.post("/render-podcast")
async def render_podcast(body: PodcastRenderRequest):
    """
    Replaces Shotstack podcast-mode assembly.
    Downloads bg loop video + audio tracks from GCS/URL, assembles with FFmpeg,
    returns MP4 binary.

    Shotstack equivalent:
      timeline.tracks[0] = background loop video (zoomInSlow effect)
      timeline.tracks[1] = host audio + reflection audio
      timeline.soundtrack = ambient music bed at low volume with fade in/out
    """
    media_temp_root = body.media_temp_root or "/mnt/HC_Volume_105210642/temp_runs"
    run_id = (body.run_id or "").strip() or uuid.uuid4().hex[:12]
    run_root = body.run_root or os.path.join(media_temp_root, run_id)
    audio_root = body.audio_root or os.path.join(run_root, "audio")
    full_audio_root = body.full_audio_root or os.path.join(audio_root, "full")
    short_audio_root = body.short_audio_root or os.path.join(audio_root, "short")
    video_root = body.video_root or os.path.join(run_root, "video")
    clips_root = body.clips_root or os.path.join(run_root, "clips")
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(audio_root, exist_ok=True)
    os.makedirs(full_audio_root, exist_ok=True)
    os.makedirs(short_audio_root, exist_ok=True)
    os.makedirs(video_root, exist_ok=True)
    os.makedirs(clips_root, exist_ok=True)
    body = body.model_copy(
        update={
            "run_root": run_root,
            "audio_root": audio_root,
            "full_audio_root": full_audio_root,
            "short_audio_root": short_audio_root,
            "video_root": video_root,
            "clips_root": clips_root,
            "run_id": run_id,
        }
    )

    job_id  = uuid.uuid4().hex[:12]
    job_dir = RENDER_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    render_ok = False
    logger.info("render-podcast job=%s run_id=%s", job_id, body.run_id)

    preset_map = {"fast": "ultrafast", "medium": "fast", "hq": "slow"}
    preset = preset_map.get(body.quality, "fast")

    # Dimensions from aspect ratio
    if body.aspect_ratio == "9:16":
        W, H = 1080, 1920
    else:
        W, H = 1920, 1080

    single_speaker = bool(body.single_speaker_mode) or int(body.cast_size or 0) <= 1
    if single_speaker:
        pruned_audio = {}
        if isinstance(body.speaker_audio, dict) and "SPEAKER_1" in body.speaker_audio:
            pruned_audio = {"SPEAKER_1": body.speaker_audio["SPEAKER_1"]}
        filtered_full_turns = [turn for turn in body.full_turns if str(turn.speaker_id or "SPEAKER_1").upper() == "SPEAKER_1"] or body.full_turns
        filtered_short_turns = [turn for turn in body.short_turns if str(turn.speaker_id or "SPEAKER_1").upper() == "SPEAKER_1"] or body.short_turns
        body = body.model_copy(update={
            "speaker_audio": pruned_audio,
            "reflection_audio_url": "",
            "reflection_audio_gcs_uri": "",
            "reflection_audio_local_path": "",
            "full_turns": filtered_full_turns,
            "short_turns": filtered_short_turns,
            "cast_size": 1,
        })

    job_token = _RENDER_JOB_DIR.set(job_dir)
    try:
        try:
            _write_render_debug_file("request.json", body.model_dump_json(indent=2))
        except Exception:
            pass

        # ── Download all assets ────────────────────────────────────────────────
        bg_path   = job_dir / "bg_loop.mp4"
        host_path = job_dir / "host.mp3"
        refl_path = job_dir / "reflection.mp3"
        out_path  = job_dir / "podcast.mp4"

        # Resolve effective title — n8n sends episode_title, model field is title
        effective_title = body.title or body.episode_title

        # ── Stage background loop (local path first, then remote URL) ───────
        bg_source_url = body.bg_loop_url or _coerce_public_storage_url(body.bg_loop_gcs_uri or body.bg_object_path)
        if not _stage_local_or_remote_asset(bg_source_url, bg_path, local_path=body.background_local_path):
            raise HTTPException(400, "Could not stage background loop video")

        def select_turns() -> list[PodcastTurn]:
            preferred = str(body.turn_scope or "full").lower().strip()
            if preferred == "short" and body.short_turns:
                return sorted(body.short_turns, key=lambda turn: turn.turn_index)
            if body.full_turns:
                return sorted(body.full_turns, key=lambda turn: turn.turn_index)
            if body.short_turns:
                return sorted(body.short_turns, key=lambda turn: turn.turn_index)
            return []

        ambient_layers = _prepare_ambient_layers(job_dir, body)
        has_ambient = bool(ambient_layers)

        selected_turns = select_turns()

        # ── Turn-based conversation path: preserve exact back-and-forth ─────
        if selected_turns:
            turn_inputs: list[str] = []
            turn_filters: list[str] = []
            mix_labels: list[str] = []
            turn_cursor = 0.0
            downloaded_turns = 0

            for local_index, turn in enumerate(selected_turns):
                turn_source_hint = turn.url or turn.gcs_uri or turn.object_path or turn.local_path
                if not turn_source_hint.strip():
                    continue
                turn_path = job_dir / _safe_turn_filename(local_index, turn.speaker_id, turn_source_hint)
                if not _stage_any_asset(
                    dest_path=turn_path,
                    local_path=turn.local_path,
                    gcs_uri=turn.gcs_uri,
                    object_path=turn.object_path,
                    url=turn.url,
                ):
                    logger.warning("job=%s skipped turn %s (%s) stage", job_id, turn.turn_index, turn.speaker_id)
                    continue

                duration = _measure_or_estimate_duration(turn_path, turn.estimated_duration)
                if duration <= 0:
                    logger.warning("job=%s skipped turn %s (%s) with unknown duration", job_id, turn.turn_index, turn.speaker_id)
                    continue

                delay_ms = max(int(round(turn_cursor * 1000)), 0)
                input_index = downloaded_turns
                label = f"turn{input_index}"
                turn_inputs += ["-i", str(turn_path)]
                turn_filters.append(
                    f"[{input_index}:a]adelay={delay_ms}|{delay_ms},volume=1.0[{label}]"
                )
                mix_labels.append(f"[{label}]")
                downloaded_turns += 1

                gap_after = max(float(turn.gap_after or 0.0), 0.0) if local_index < len(selected_turns) - 1 else 0.0
                turn_cursor += duration + gap_after

            if not downloaded_turns:
                raise HTTPException(502, "Turn audio was provided but none of the turn clips could be downloaded")

            total = max(turn_cursor + 0.5, 0.5)
            if len(mix_labels) > 1:
                turn_filters.append(
                    f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0[dialogue]"
                )
                dialogue_label = "dialogue"
            else:
                single_label = mix_labels[0].strip("[]")
                if single_label != "dialogue":
                    turn_filters.append(f"[{single_label}]anull[dialogue]")
                dialogue_label = "dialogue"

            audio_map_label = f"[{dialogue_label}]"
            if has_ambient:
                for ambient_index, layer in enumerate(ambient_layers):
                    layer["input_index"] = downloaded_turns + ambient_index
                    turn_inputs += [
                        "-stream_loop", "-1", "-t", f"{total:.2f}", "-i", str(layer["path"])
                    ]
                audio_map_label = _append_ambient_layers_mix(
                    turn_filters,
                    dialogue_label,
                    dialogue_label,
                    ambient_layers,
                    total,
                )

            mixed_path = job_dir / "mixed.aac"
            _run_ffmpeg(
                turn_inputs + [
                    "-filter_complex", ";".join(turn_filters),
                    "-map", audio_map_label,
                    "-c:a", "aac", "-b:a", "192k", "-vn",
                    str(mixed_path),
                ]
            )

            logger.info(
                "job=%s interleaved dialogue: %d turns, total=%.1fs, style=%s ambient_layers=%d ducking=%s",
                job_id,
                downloaded_turns,
                total,
                body.render_style,
                len(ambient_layers),
                body.ambient_ducking,
            )

        # ── Speaker fallback: flatten one file per speaker in fixed order ────
        # This keeps older workflows working, but it does not preserve turn order.
        elif body.speaker_audio:
            SPEAKER_ORDER = ["SPEAKER_1"] if single_speaker else [
                "SPEAKER_1",
                "SPEAKER_2",
                "SPEAKER_3",
                "SPEAKER_4",
                "SPEAKER_GIRL",
                "SPEAKER_GIRL_2",
            ]
            spk_entries = [
                body.speaker_audio[k]
                for k in SPEAKER_ORDER
                if k in body.speaker_audio and _has_speaker_scope_source(body.speaker_audio[k], "full")
            ]
            if not spk_entries:
                raise HTTPException(400, "speaker_audio present but no usable full speaker sources found")
            spk_paths = []
            for idx, entry in enumerate(spk_entries):
                p = job_dir / f"spk_{idx}.mp3"
                if _stage_any_asset(
                    dest_path=p,
                    local_path=_speaker_scope_value(entry, "full", "local_path"),
                    gcs_uri=_speaker_scope_value(entry, "full", "gcs_uri"),
                    object_path=_speaker_scope_value(entry, "full", "object_path"),
                    url=_speaker_scope_value(entry, "full", "url"),
                ):
                    spk_paths.append(p)
            if not spk_paths:
                raise HTTPException(502, "Failed to stage any speaker audio files")
            if len(spk_paths) == 1:
                import shutil as _sh2; _sh2.copy(str(spk_paths[0]), str(host_path))
            else:
                spk_inputs = []
                for p in spk_paths:
                    spk_inputs += ["-i", str(p)]
                n = len(spk_paths)
                fc = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[outa]"
                _run_ffmpeg(spk_inputs + [
                    "-filter_complex", fc,
                    "-map", "[outa]",
                    "-codec:a", "libmp3lame", "-q:a", "2",
                    str(host_path),
                ])
            if not host_path.exists() or host_path.stat().st_size == 0:
                raise HTTPException(500, "Multi-speaker concat produced empty file")
            # Reflection track already baked in — disable separate refl download
            body = body.model_copy(update={"reflection_audio_url": ""})
            logger.info("job=%s multi-speaker concat: %d tracks -> host.mp3", job_id, len(spk_paths))

        # ── Legacy 2-track path ──────────────────────────────────────────────
        elif body.host_audio_url or body.host_audio_local_path or body.host_audio_gcs_uri:
            if not _stage_any_asset(
                dest_path=host_path,
                local_path=body.host_audio_local_path,
                gcs_uri=body.host_audio_gcs_uri,
                url=body.host_audio_url,
            ):
                raise HTTPException(400, "Could not stage host audio")
        else:
            raise HTTPException(400, "Provide full_turns/short_turns, speaker_audio, or host_audio_url")

        if not selected_turns:
            has_reflection = any([
                str(body.reflection_audio_url or "").strip(),
                str(body.reflection_audio_local_path or "").strip(),
                str(body.reflection_audio_gcs_uri or "").strip(),
            ])

            if has_reflection:
                has_reflection = _stage_any_asset(
                    dest_path=refl_path,
                    local_path=body.reflection_audio_local_path,
                    gcs_uri=body.reflection_audio_gcs_uri,
                    url=body.reflection_audio_url,
                )

            # ── Measure durations ─────────────────────────────────────────────
            host_dur = body.host_duration or _get_audio_duration(host_path)
            refl_dur = (body.reflection_duration or _get_audio_duration(refl_path)) if has_reflection else 0.0
            total = host_dur + (refl_dur + body.gap_seconds if has_reflection else 0) + 0.5

            logger.info("job=%s host=%.1fs refl=%.1fs total=%.1fs", job_id, host_dur, refl_dur, total)

            # ── Pass 1: audio-only mix (re-indexed, no video, no loudnorm) ──
            audio_mix_inputs = ["-i", str(host_path)]
            audio_fc_parts = ["[0:a]volume=1.0[host]"]
            mix_labels = ["[host]"]
            mix_count = 1
            audio_idx = 1

            if has_reflection:
                delay_ms = int((host_dur + body.gap_seconds) * 1000)
                audio_mix_inputs += ["-i", str(refl_path)]
                audio_fc_parts.append(
                    f"[{audio_idx}:a]adelay={delay_ms}|{delay_ms},volume=1.0[refl]"
                )
                mix_labels.append("[refl]")
                mix_count += 1
                audio_idx += 1

            if mix_count > 1:
                audio_fc_parts.append(
                    f"{''.join(mix_labels)}amix=inputs={mix_count}:duration=longest:dropout_transition=2[dialogue]"
                )
                dialogue_label = "dialogue"
            else:
                audio_fc_parts.append("[host]anull[dialogue]")
                dialogue_label = "dialogue"

            audio_map_label = f"[{dialogue_label}]"
            if has_ambient:
                for ambient_index, layer in enumerate(ambient_layers):
                    layer["input_index"] = audio_idx + ambient_index
                    audio_mix_inputs += [
                        "-stream_loop", "-1", "-t", f"{total:.2f}", "-i", str(layer["path"])
                    ]
                audio_map_label = _append_ambient_layers_mix(
                    audio_fc_parts,
                    dialogue_label,
                    dialogue_label,
                    ambient_layers,
                    total,
                )

            mixed_path = job_dir / "mixed.aac"
            _run_ffmpeg(
                audio_mix_inputs + [
                    "-filter_complex", ";".join(audio_fc_parts),
                    "-map", audio_map_label,
                    "-c:a", "aac", "-b:a", "192k", "-vn",
                    str(mixed_path),
                ]
            )

        # ── Pass 2: normalize the mixed audio ────────────────────────────────
        normed_path = _prenorm_audio(mixed_path, job_dir / "normed.aac")

        # ── Optional subtitles from the mixed dialogue/program audio ───────
        srt_path: Path | None = None
        if body.show_subtitles:
            try:
                srt_path = _generate_srt(
                    mixed_path,
                    job_dir,
                    caption_mode=body.caption_mode,
                    caption_hook_mode=body.caption_hook_mode,
                    caption_landing_mode=body.caption_landing_mode,
                    max_words_per_caption=body.max_words_per_caption,
                    max_chars_per_line=body.max_chars_per_line,
                    max_lines=body.max_lines,
                    portrait_safe_margin=body.portrait_safe_margin,
                    caption_vertical_zone=body.caption_vertical_zone,
                    dynamic_word_emphasis=body.dynamic_word_emphasis,
                    caption_animation_level=body.caption_animation_level,
                    caption_density_hint=body.caption_density_hint,
                    render_width=W,
                    render_height=H,
                )
            except Exception as e:
                logger.warning("job=%s subtitle generation failed: %s", job_id, e)
                srt_path = None

        # ── Rotating backgrounds: use synthesized sequence when multiple locals exist ─
        bg_sequence_path = _build_background_sequence(
            body.background_local_paths, total, job_dir, W, H, preset
        )
        effective_bg_path = bg_sequence_path or bg_path

        # ── Pass 3: video render with pre-normed audio (waveform + captions) ─
        inputs = [
            "-stream_loop",
            "-1",
            "-t",
            f"{total:.2f}",
            "-i",
            str(effective_bg_path),
            "-i",
            str(normed_path),
        ]

        wave_h = 220 if H >= 1600 else max(140, int(H * 0.18))
        wave_top = H - wave_h - 220 if H >= 1600 else H - wave_h - 140
        title_y = 140 if H >= 1600 else 80
        scripture_y = H - 120 if H >= 1600 else H - 80

        # Still image → Ken Burns zoom; real video → preserve motion (avoid zoompan freeze bug).
        bg_is_still = effective_bg_path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        }
        if bg_is_still:
            video_filter = (
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},{_WWM_VIDEO_SHARPEN},"
                f"zoompan=z='min(zoom+0.0003,1.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={int(total * 30)}:s={W}x{H}:fps=30[bgvid]"
            )
        else:
            video_filter = (
                f"[0:v]fps=30,"
                f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},"
                f"{_WWM_VIDEO_SHARPEN},"
                f"setpts=PTS-STARTPTS[bgvid]"
            )

        def esc(t: str) -> str:
            return str(t).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

        filter_parts = [video_filter]
        current_video = "bgvid"
        current_audio = "1:a"

        viz_style = str(body.audio_visual_style or "waveform").lower().strip()
        if viz_style in ("waveform", "spectrum"):
            if viz_style == "spectrum":
                viz = (
                    f"[1:a]asplit=2[a_vis][a_enc];"
                    f"[a_vis]showspectrum=s={W}x{wave_h}:mode=combined:slide=scroll:"
                    f"color=rainbow:scale=cbrt[viz]"
                )
            else:
                viz = (
                    f"[1:a]asplit=2[a_vis][a_enc];"
                    f"[a_vis]showwaves=s={W}x{wave_h}:mode=cline:"
                    f"colors=0xE8A838|0xE8A838:rate=30[viz]"
                )
            filter_parts.append(viz)
            filter_parts.append(f"[{current_video}][viz]overlay=0:{wave_top}[v_wave]")
            current_video = "v_wave"
            current_audio = "[a_enc]"

        vignette_y = int(H * 0.55)
        vignette_h = H - vignette_y
        filter_parts.append(
            f"[{current_video}]drawbox=x=0:y={vignette_y}:w={W}:h={vignette_h}:color=black@0.22:t=fill,"
            f"drawbox=x=0:y={int(H * 0.68)}:w={W}:h={H - int(H * 0.68)}:color=black@0.10:t=fill[v_vignette]"
        )
        current_video = "v_vignette"

        overlays = []
        if body.show_title and effective_title:
            overlays.append(
                f"drawtext=text='{esc(effective_title[:60])}':fontsize=44:fontcolor=0xF5F0E8:"
                f"x=(w-text_w)/2:y={title_y}:shadowx=0:shadowy=2:shadowcolor=black@0.55:"
                f"alpha='if(lt(t,0.5),t/0.5,if(lt(t,3.0),1,max(0,1-(t-3.0)/0.5)))'"
            )
        if body.scripture:
            scripture_fade_out = max(2.8, total - 0.8)
            overlays.append(
                f"drawtext=text='{esc(body.scripture[:100])}':fontsize=38:fontcolor=0xE8A838:"
                f"x=(w-text_w)/2:y={scripture_y}:shadowx=0:shadowy=2:shadowcolor=black@0.60:"
                f"alpha='if(lt(t,0.6),0,if(lt(t,1.2),(t-0.6)/0.6,"
                f"if(lt(t,{scripture_fade_out:.2f}),1,max(0,1-(t-{scripture_fade_out:.2f})/0.5))))'"
            )
        fade_v_st = max(0.0, total - 0.45)
        fade_v = f"fade=t=in:st=0:d=0.4,fade=t=out:st={fade_v_st:.2f}:d=0.45"
        if overlays:
            filter_parts.append(f"[{current_video}]" + ",".join(overlays) + f",{fade_v}[v_text]")
            current_video = "v_text"
        else:
            filter_parts.append(f"[{current_video}]{fade_v}[v_text]")
            current_video = "v_text"

        if srt_path:
            fs = "" if srt_path.suffix.lower() == ".ass" else _srt_force_style(body.caption_style)
            sub_clause = _subtitle_filter_clause(srt_path, fs)
            filter_parts.append(f"[{current_video}]{sub_clause}[vout]")
            current_video = "vout"
        else:
            filter_parts.append(f"[{current_video}]null[vout]")
            current_video = "vout"

        _run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filter_parts),
                "-map", f"[{current_video}]",
                "-map", current_audio,
                "-c:v", "libx264", "-preset", preset, "-crf", "22",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-t", f"{total:.2f}",
                str(out_path),
            ]
        )

        logger.info("job=%s podcast render complete size=%d", job_id, out_path.stat().st_size)
        render_ok = True
        return FileResponse(
            str(out_path), media_type="video/mp4",
            filename=f"wwm_podcast_{job_id}.mp4",
            headers={"X-Job-Id": job_id, "X-Duration": f"{total:.1f}"},
        )

    except HTTPException as e:
        _write_render_debug_file(
            "http_exception.txt",
            str(e.detail if hasattr(e, "detail") else e),
        )
        raise
    except RuntimeError as e:
        logger.error("job=%s podcast FFmpeg error: %s", job_id, str(e)[:1200])
        _write_render_debug_file("render_error.txt", str(e))
        raise HTTPException(500, f"Podcast render failed: {str(e)[:1200]}")
    except Exception as e:
        logger.error("job=%s podcast unexpected: %s", job_id, e, exc_info=True)
        _write_render_debug_file("render_error.txt", str(e))
        raise HTTPException(500, str(e))
    finally:
        _RENDER_JOB_DIR.reset(job_token)
        def _cleanup():
            time.sleep(90 if render_ok else 21600)
            shutil.rmtree(job_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()


# ── /render-video-clips — Shotstack replacement (Veo clip concat mode) ────────
# Concatenates AI-generated video clips (Veo 3 / any MP4) + overlays host audio
# ──────────────────────────────────────────────────────────────────────────────

class VideoClipsRenderRequest(BaseModel):
    clip_urls:      list[str] = Field(..., description="Ordered list of MP4 clip URLs (Veo GCS URLs)")
    host_audio_url: str       = Field(..., description="GCS URL of host MP3")
    ambient_music_url: str    = Field("",  description="Optional ambient music URL")
    ambient_volume: float     = Field(0.08)
    aspect_ratio:   str       = Field("9:16")
    crossfade_secs: float     = Field(0.5, description="Dissolve duration between clips")
    quality:        str       = Field("medium")


@app.post("/render-video-clips")
async def render_video_clips(body: VideoClipsRenderRequest):
    """
    Replaces Shotstack video-mode assembly (used with Veo 3 AI clips).
    Downloads clips + audio from GCS, concatenates with crossfades, returns MP4.
    """
    if not body.clip_urls:
        raise HTTPException(400, "clip_urls must not be empty")

    job_id  = uuid.uuid4().hex[:12]
    job_dir = RENDER_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_token = _RENDER_JOB_DIR.set(job_dir)
    logger.info("render-video-clips job=%s clips=%d", job_id, len(body.clip_urls))

    preset_map = {"fast": "ultrafast", "medium": "fast", "hq": "slow"}
    preset = preset_map.get(body.quality, "fast")
    W, H   = (1080, 1920) if body.aspect_ratio == "9:16" else (1920, 1080)

    try:
        # ── Download clips ────────────────────────────────────────────────────
        clip_paths: list[Path] = []
        for i, url in enumerate(body.clip_urls):
            dest = job_dir / f"clip_{i:03d}.mp4"
            if _download_file(url, dest):
                clip_paths.append(dest)
            else:
                logger.warning("job=%s clip %d download failed, skipping", job_id, i)

        if not clip_paths:
            raise HTTPException(400, "No clips could be downloaded")

        # ── Download audio ────────────────────────────────────────────────────
        host_path = job_dir / "host.mp3"
        amb_path  = job_dir / "ambient.mp3"
        if not _download_file(body.host_audio_url, host_path):
            raise HTTPException(400, "Could not download host audio")

        has_ambient = bool(body.ambient_music_url.strip())
        if has_ambient:
            _download_file(body.ambient_music_url, amb_path)
            has_ambient = amb_path.exists() and amb_path.stat().st_size > 0

        host_dur = _get_audio_duration(host_path)
        fade_out_start = max(0.0, host_dur - 3.0)

        out_path = job_dir / "clips_render.mp4"

        # ── Pre-normalize audio (isolated pass keeps loudnorm out of filter_complex) ─
        if has_ambient:
            # Mix host + ambient first, then normalize
            mixed_path = job_dir / "mixed.aac"
            _run_ffmpeg([
                "-i", str(host_path),
                "-stream_loop", "-1", "-i", str(amb_path),
                "-filter_complex",
                (
                    f"[1:a]volume={body.ambient_volume},"
                    f"afade=t=in:st=0:d=2,"
                    f"afade=t=out:st={fade_out_start:.2f}:d=3[amb];"
                    f"[0:a][amb]amix=inputs=2:duration=first:dropout_transition=2[amix]"
                ),
                "-map", "[amix]",
                "-c:a", "aac", "-b:a", "192k", "-vn",
                str(mixed_path),
            ])
            normed_path = _prenorm_audio(mixed_path, job_dir / "normed.aac")
        else:
            normed_path = _prenorm_audio(host_path, job_dir / "normed.aac")

        # ── Build filter_complex for clip concat with crossfades ──────────────
        n = len(clip_paths)
        inputs = []
        for p in clip_paths:
            inputs += ["-i", str(p)]
        inputs += ["-i", str(normed_path)]   # audio at index n (pre-normed, no ambient needed)

        audio_idx  = n       # normed audio input index

        filter_parts = []

        # Scale each clip to target resolution (+ light sharpen)
        for i in range(n):
            filter_parts.append(
                f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},{_WWM_VIDEO_SHARPEN},setsar=1[v{i}]"
            )

        # Crossfade chain between clips
        if n == 1:
            filter_parts.append(f"[v0]copy[vout]")
        else:
            # Accumulate clip durations for crossfade offsets
            clip_durs = [_get_audio_duration(p) for p in clip_paths]
            prev  = "v0"
            offset = 0.0
            for i in range(1, n):
                offset += clip_durs[i-1] - body.crossfade_secs
                out_lbl = f"xf{i}" if i < n - 1 else "vout"
                filter_parts.append(
                    f"[{prev}][v{i}]xfade=transition=dissolve"
                    f":duration={body.crossfade_secs}:offset={offset:.2f}[{out_lbl}]"
                )
                prev = f"xf{i}"

        # Audio: pre-normed audio passed through directly (mixing/normalization
        # already done in the dedicated pre-norm pass above).
        filter_parts.append(f"[{audio_idx}:a]acopy[afinal]")

        clip_fade_out = max(0.0, host_dur - 0.45)
        filter_parts.append(
            f"[vout]fade=t=in:st=0:d=0.35,"
            f"fade=t=out:st={clip_fade_out:.2f}:d=0.45[vfinal]"
        )

        filter_complex = ";".join(filter_parts)

        _run_ffmpeg(
            inputs + [
                "-filter_complex", filter_complex,
                "-map", "[vfinal]",
                "-map", "[afinal]",
                "-c:v", "libx264", "-preset", preset, "-crf", "22",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(out_path),
            ]
        )

        logger.info("job=%s clips render complete size=%d", job_id, out_path.stat().st_size)
        return FileResponse(
            str(out_path), media_type="video/mp4",
            filename=f"wwm_veo_{job_id}.mp4",
            headers={"X-Job-Id": job_id, "X-Clip-Count": str(n)},
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("job=%s clips FFmpeg error: %s", job_id, str(e)[:500])
        raise HTTPException(500, f"Clip render failed: {str(e)[:400]}")
    except Exception as e:
        logger.error("job=%s clips unexpected: %s", job_id, e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        _RENDER_JOB_DIR.reset(job_token)
        def _cleanup():
            time.sleep(90)
            shutil.rmtree(job_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()
