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
import uuid
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
    logger.info("FFmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])
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
        # Clean up job folder after response is sent
        # (FileResponse streams before this runs, so the file is safe)
        import threading
        def _cleanup():
            import time
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


def _generate_srt(audio_path: Path, job_dir: Path) -> Path | None:
    """Run Whisper on audio and write a properly formatted SRT file."""
    try:
        model = get_whisper()
        segments, _ = model.transcribe(
            str(audio_path), beam_size=5, vad_filter=True,
        )
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = _seconds_to_srt_time(seg.start)
            end   = _seconds_to_srt_time(seg.end)
            text  = seg.text.strip()
            if text:
                srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        if not srt_lines:
            return None
        srt_path = job_dir / "captions.srt"
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        return srt_path
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

    url = _coerce_public_storage_url(
        payload.get("url")
        or payload.get("src")
        or payload.get("ambient_music_url")
        or payload.get("music_gcs_uri")
        or payload.get("gcs_uri")
        or payload.get("gcs_url")
    )
    if not url:
        return None

    def pick(key: str, fallback):
        value = payload.get(key)
        return fallback if value is None else value

    return {
        "name": str(payload.get("name") or default_name),
        "url": url,
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
            "volume": body.ambient_volume,
            "ducking": body.ambient_ducking,
            "duck_threshold": body.ambient_duck_threshold,
            "duck_ratio": body.ambient_duck_ratio,
            "duck_attack": body.ambient_duck_attack,
            "duck_release": body.ambient_duck_release,
            "duck_makeup": body.ambient_duck_makeup,
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
        ext = Path(layer["url"].split("?", 1)[0]).suffix or ".mp3"
        slug = re.sub(r"[^a-z0-9_-]+", "_", str(layer["name"]).lower()).strip("_") or f"ambient_{idx + 1}"
        path = job_dir / f"ambient_{idx + 1:02d}_{slug}{ext}"
        if not _download_file(layer["url"], path):
            logger.warning("ambient layer download failed name=%s url=%s", layer["name"], layer["url"])
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
    """Return an ASS/SRT force_style string for the subtitles FFmpeg filter."""
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
    return base   # default: bottom, Alignment=2


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
        return t.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")

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
        import threading
        def _cleanup():
            import time; time.sleep(90)
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
    bg_loop_url:        str   = Field(...,  description="GCS URL of background loop video")
    host_audio_url:     str   = Field("",   description="GCS URL of host MP3 (empty when speaker_audio used)")
    reflection_audio_url: str = Field("",   description="GCS URL of reflection MP3 (optional)")
    # Multi-speaker fields sent by n8n Cast TTS node
    speaker_audio:      dict  = Field(default_factory=dict, description="SPEAKER_1..N dict from Cast TTS")
    full_turns:         list[PodcastTurn] = Field(default_factory=list, description="Full episode turn clips in playback order")
    short_turns:        list[PodcastTurn] = Field(default_factory=list, description="Short-cut turn clips in playback order")
    full_sequence:      list[dict] = Field(default_factory=list, description="Raw full combined_sequence from the writer")
    short_sequence:     list[dict] = Field(default_factory=list, description="Raw short combined_sequence from the writer")
    render_style:       str   = Field("speaker_fallback", description="interleaved_dialogue | speaker_fallback")
    turn_scope:         str   = Field("full", description="Prefer full or short turns when both are present")
    cast_size:          int   = Field(0,    description="Number of speakers (0 = derive from speaker_audio)")
    run_id:             str   = Field("",   description="Episode run_id from n8n")
    episode_title:      str   = Field("",   description="Alias for title (n8n sends episode_title)")
    ambient_music_url:  str   = Field("",  description="GCS URL or Pixabay URL of ambient music bed")
    music_gcs_uri:      str   = Field("",  description="Legacy gs:// music URI fallback")
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
    aspect_ratio:       str   = Field("9:16", description="'9:16' (Reels/Shorts) or '16:9' (YouTube)")
    title:              str   = Field("",  description="Episode title for drawtext overlay")
    scripture:          str   = Field("",  description="Scripture reference for drawtext overlay")
    quality:            str   = Field("medium", description="fast | medium | hq")


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
    job_id  = uuid.uuid4().hex[:12]
    job_dir = RENDER_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    logger.info("render-podcast job=%s", job_id)

    preset_map = {"fast": "ultrafast", "medium": "fast", "hq": "slow"}
    preset = preset_map.get(body.quality, "fast")

    # Dimensions from aspect ratio
    if body.aspect_ratio == "9:16":
        W, H = 1080, 1920
    else:
        W, H = 1920, 1080

    try:
        # ── Download all assets ────────────────────────────────────────────────
        bg_path   = job_dir / "bg_loop.mp4"
        host_path = job_dir / "host.mp3"
        refl_path = job_dir / "reflection.mp3"
        out_path  = job_dir / "podcast.mp4"

        # Resolve effective title — n8n sends episode_title, model field is title
        effective_title = body.title or body.episode_title

        # ── Download background loop ─────────────────────────────────────────
        if not _download_file(body.bg_loop_url, bg_path):
            raise HTTPException(400, "Could not download background loop video")

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
                if not turn.url.strip():
                    continue
                turn_path = job_dir / _safe_turn_filename(local_index, turn.speaker_id, turn.url)
                if not _download_file(turn.url, turn_path):
                    logger.warning("job=%s skipped turn %s (%s) download", job_id, turn.turn_index, turn.speaker_id)
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
                dialogue_label = mix_labels[0].strip("[]")

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
            SPEAKER_ORDER = ["SPEAKER_1", "SPEAKER_2", "SPEAKER_3", "SPEAKER_4", "SPEAKER_GIRL"]
            spk_urls = [
                body.speaker_audio[k]["full_url"]
                for k in SPEAKER_ORDER
                if k in body.speaker_audio and body.speaker_audio[k].get("full_url")
            ]
            if not spk_urls:
                raise HTTPException(400, "speaker_audio present but no full_url entries found")
            spk_paths = []
            for idx, url in enumerate(spk_urls):
                p = job_dir / f"spk_{idx}.mp3"
                if _download_file(url, p):
                    spk_paths.append(p)
            if not spk_paths:
                raise HTTPException(502, "Failed to download any speaker audio files")
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
        elif body.host_audio_url:
            if not _download_file(body.host_audio_url, host_path):
                raise HTTPException(400, "Could not download host audio")
        else:
            raise HTTPException(400, "Provide full_turns/short_turns, speaker_audio, or host_audio_url")

        if not selected_turns:
            has_reflection = bool(body.reflection_audio_url.strip())

            if has_reflection:
                _download_file(body.reflection_audio_url, refl_path)
                has_reflection = refl_path.exists() and refl_path.stat().st_size > 0

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
                dialogue_label = "host"

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

        # ── Pass 3: video render with pre-normed audio (no loudnorm in graph) ─
        inputs = [
            "-stream_loop", "-1", "-t", f"{total:.2f}", "-i", str(bg_path),   # 0: bg video
            "-i", str(normed_path),                                             # 1: normed audio
        ]

        # ── Build filter_complex (video only) ────────────────────────────────
        # Video: scale bg loop to fit frame, slow Ken Burns zoom in + light sharpen
        video_filter = (
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},{_WWM_VIDEO_SHARPEN},"
            f"zoompan=z='min(zoom+0.0003,1.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={int(total*30)}:s={W}x{H}:fps=30[bgvid]"
        )

        # Optional text overlays (episode title top, scripture bottom)
        def esc(t: str) -> str:
            return t.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")

        overlays = []
        if effective_title:
            overlays.append(
                f"drawtext=text='{esc(effective_title[:60])}':fontsize=52:fontcolor=0xF5F0E8:"
                f"x=(w-text_w)/2:y=140:"
                f"alpha='if(lt(t,0.5),t/0.5,if(lt(t,{total-1:.1f}),1,max(0,1-(t-{total-1:.1f})/0.5)))'"
            )
        if body.scripture:
            overlays.append(
                f"drawtext=text='{esc(body.scripture[:100])}':fontsize=36:fontcolor=0xE8A838:"
                f"x=(w-text_w)/2:y=h-120:"
                f"alpha='if(lt(t,1),0,if(lt(t,2),(t-1),1))'"
            )

        fade_v_st = max(0.0, total - 0.45)
        fade_v = f"fade=t=in:st=0:d=0.4,fade=t=out:st={fade_v_st:.2f}:d=0.45"
        if overlays:
            video_chain = "[bgvid]" + ",".join(overlays) + f",{fade_v}[vout]"
        else:
            video_chain = f"[bgvid]{fade_v}[vout]"

        _run_ffmpeg(
            inputs + [
                "-filter_complex", f"{video_filter};{video_chain}",
                "-map", "[vout]",
                "-map", "1:a",
                "-c:v", "libx264", "-preset", preset, "-crf", "22",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-t", f"{total:.2f}",
                str(out_path),
            ]
        )

        logger.info("job=%s podcast render complete size=%d", job_id, out_path.stat().st_size)
        return FileResponse(
            str(out_path), media_type="video/mp4",
            filename=f"wwm_podcast_{job_id}.mp4",
            headers={"X-Job-Id": job_id, "X-Duration": f"{total:.1f}"},
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("job=%s podcast FFmpeg error: %s", job_id, str(e)[:500])
        raise HTTPException(500, f"Podcast render failed: {str(e)[:400]}")
    except Exception as e:
        logger.error("job=%s podcast unexpected: %s", job_id, e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        import threading
        def _cleanup():
            import time; time.sleep(90)
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
        import threading
        def _cleanup():
            import time; time.sleep(90)
            shutil.rmtree(job_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()
