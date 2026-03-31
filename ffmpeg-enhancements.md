# FFmpeg enhancements (change log)

Summary of the processing added in `app/main.py`: loudness normalization, peak limiting, light sharpening, and full-frame video fades.

## New helpers and constants

| Symbol | Purpose |
|--------|---------|
| `_WWM_AUDIO_MASTER` | `loudnorm=I=-14:TP=-1.0:LRA=11` plus `alimiter=limit=0.98:attack=2:release=50` — streaming/social-style integrated loudness (~−14 LUFS) and true-peak safety. |
| `_WWM_VIDEO_SHARPEN` | `unsharp=5:5:0.65:3:3:0.0` after scale/crop to reduce softness from upscaling. |
| `_filter_audio_master(in_label, out_label)` | Builds one filter-graph segment: `in_label` + `_WWM_AUDIO_MASTER` + `out_label` (e.g. `[aout]` → `[afinal]`). |

Location: immediately after `_run_ffmpeg()` in `app/main.py`.

## Audio behavior

- **Final encode only:** Mastering runs on the audio that is muxed into the output MP4. `_mix_audio()` (voice + Pixabay bed) is **unchanged**, so dynamic renders are not loud-normalized twice.
- **Waveform/spectrum paths:** Input audio is split with `asplit=2`: one branch drives `showwaves` / `showspectrum`, the other goes through `_filter_audio_master` to `[a_enc]` (FFmpeg requires a split when one stream feeds two filters).

## Video behavior

- **Sharpen:** Inserted in the filter chain after `crop` (and before `zoompan` where applicable) for podcast background, slideshow slides, and Veo clip scaling.
- **Fades:** Short `fade` in/out on the composed video where noted below (timings are approximate).

## Endpoints and functions touched

| Area | Changes |
|------|---------|
| `POST /render-audiogram` | `asplit`; map `[a_enc]` instead of `0:a`; `_filter_audio_master('[a_src]', '[a_enc]')`. |
| `_fmt_audiogram` | Same as audiogram (used by `/render-dynamic`). |
| `_fmt_captioned` | Same `asplit` + mastered audio map. |
| `_fmt_scripture_cards` | Same `asplit` + mastered audio map. |
| `_fmt_slideshow` | `_WWM_VIDEO_SHARPEN` in each slide chain; video fade in/out vs. voice duration; `[vfinal]` + `_filter_audio_master("[0:a]", "[aenc]")`. |
| `POST /render-podcast` | Sharpen after crop on background loop; drawtext chain (if any) then video fade; mix → `[aout]` → `_filter_audio_master` → `[afinal]`; always map `[vout]` + `[afinal]`. |
| `POST /render-video-clips` | Sharpen on scaled clips; host or host+ambient → `[afinal]`; `[vout]` → fade → `[vfinal]`; map `[vfinal]` + `[afinal]`. |

## Unchanged

- `_fmt_full` still concatenates intro + main with stream copy; each segment is already mastered by the underlying `_fmt_*` calls.
- `POST /youtube-context` / `yt-dlp` post-processing flags are unchanged.

## Operations notes

- **CPU cost:** `loudnorm` adds work per render; expect slightly longer encodes vs. raw AAC.
- **Environment:** Local Windows shells may not have `ffmpeg` on `PATH`; the project `Dockerfile` installs Debian’s `ffmpeg`, which includes these filters.

## Adjusting behavior

To change targets or strength, edit `_WWM_AUDIO_MASTER` and `_WWM_VIDEO_SHARPEN` (or refactor them to read from `os.environ`) in `app/main.py` next to `_filter_audio_master`.
