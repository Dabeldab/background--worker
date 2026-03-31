# n8n Workflow Changes — Dynamic Video Rendering + Shotstack Replacement
# Apply to DV001_Daily_Devotional_Orchestrator_v17_FIXED.json

═══════════════════════════════════════════════════════════════════════
PART 1 — REPLACE SHOTSTACK WITH HETZNER WORKER
═══════════════════════════════════════════════════════════════════════

Shotstack is now optional. Your Hetzner worker has equivalent endpoints.
Cost savings: ~$10-15 per episode in Shotstack render fees → $0.

CHANGE 1 — Set Global Config
─────────────────────────────
Change assembly_mode from "shotstack" to "background_worker"
(Keep "audio_only" and "shotstack" as commented fallback options)

  assembly_mode: "background_worker"


CHANGE 2 — Assembly Mode Router
─────────────────────────────────
Current outputs:
  output[0] → 🎧 Extract Audio URL1        (audio_only mode)
  output[1] → 🎞️ Build Assembly Payload   (shotstack mode)

Add output[2]:
  Condition: assembly_mode === 'background_worker'
  output[2] → 🎬 Render Podcast Video      (new node below)


CHANGE 3 — Replace "🎬 Shotstack: Submit Assembly" chain
──────────────────────────────────────────────────────────
DISABLE these 4 nodes (they are the Shotstack polling chain):
  ⏳ Wait for Shotstack (60s)
  🔍 Poll Shotstack Status
  ✅ Assembly Ready?
  ⏳ Wait 60s More (Assembly)

ADD this single node instead:

  Node: 🎬 Render Podcast Video
  Type: n8n-nodes-base.httpRequest
  Method: POST
  URL: {{ $env.BACKGROUND_WORKER_URL }}/render-podcast
  Body type: application/json
  JSON body:
  {
    "bg_loop_url":           "={{ $('🌅 Select Background Loop').first().json.loop_url }}",
    "host_audio_url":        "={{ 'https://storage.googleapis.com/' + $('Set Global Config').first().json.gcs_bucket + '/runs/' + $('Set Global Config').first().json.run_id + '/audio/host.mp3' }}",
    "reflection_audio_url":  "={{ 'https://storage.googleapis.com/' + $('Set Global Config').first().json.gcs_bucket + '/runs/' + $('Set Global Config').first().json.run_id + '/audio/reflection.mp3' }}",
    "ambient_music_url":     "={{ 'https://storage.googleapis.com/' + $('Set Global Config').first().json.gcs_bucket + '/music/ambient_bed.mp3' }}",
    "ambient_volume":        0.08,
    "aspect_ratio":          "={{ $('Set Global Config').first().json.aspect_ratio || '9:16' }}",
    "title":                 "={{ $('✍️ Voice Writer Agent').first().json.output.episode_title || '' }}",
    "scripture":             "={{ $('🌐 Merge Signal Source').first().json.world_pulse?.calendar?.primary_verse || '' }}",
    "quality":               "medium"
  }
  Response format: file
  Output property: video
  Timeout: 300000 (5 min)
  continueOnFail: true

  → connects to: ☁️ Upload Rendered Podcast to GCS  (new node below)


  Node: ☁️ Upload Rendered Podcast to GCS
  Type: n8n-nodes-base.httpRequest
  Method: POST
  URL: https://storage.googleapis.com/upload/storage/v1/b/{{ $('Set Global Config').first().json.gcs_bucket }}/o?uploadType=media&name=runs/{{ $('Set Global Config').first().json.run_id }}/video/episode.mp4
  Headers:
    Content-Type: video/mp4
    Authorization: Bearer {{ $('Set Global Config').first().json.gcs_token }}
  Body: binary, field name: video
  Send binary data: true

  → connects to: 🎥 Extract Final Video URL


CHANGE 4 — Extract Final Video URL (add background_worker branch)
──────────────────────────────────────────────────────────────────
In the jsCode, find the else block and extend it:

  } else if (assemblyMode === 'background_worker') {
    const gcs = (() => {
      try { return $('☁️ Upload Rendered Podcast to GCS').first().json; } catch(e) { return {}; }
    })();
    finalUrl = gcs.mediaLink || gcs.selfLink || '';
  }


CHANGE 5 — Shotstack Short Cut (optional replacement)
───────────────────────────────────────────────────────
The Short Cut flow also uses Shotstack. You can disable:
  🎞️ Build Short Cut Payload
  🎬 Shotstack: Submit Short Cut
  (and any polling nodes for short cut)

And replace with a second call to /render-podcast using the short_cut GCS audio URLs:
  host_audio_url:       runs/{run_id}/audio/host_short.mp3
  reflection_audio_url: runs/{run_id}/audio/reflection_short.mp3
  (same bg_loop_url and ambient)


═══════════════════════════════════════════════════════════════════════
PART 2 — REPLACE SHOTSTACK VIDEO MODE (Veo clips)
═══════════════════════════════════════════════════════════════════════

When assembly_mode is NOT podcast (i.e. Veo 3 scene clips are used):
Replace the Shotstack render with /render-video-clips

  Node: 🎬 Render Veo Clips
  Type: n8n-nodes-base.httpRequest
  Method: POST
  URL: {{ $env.BACKGROUND_WORKER_URL }}/render-video-clips
  Body type: application/json
  JSON body:
  {
    "clip_urls":       "={{ $('📦 Fetch All Clips for Assembly').all().map(i => i.json.video_url) }}",
    "host_audio_url":  "={{ 'https://storage.googleapis.com/' + $('Set Global Config').first().json.gcs_bucket + '/runs/' + $('Set Global Config').first().json.run_id + '/audio/host.mp3' }}",
    "ambient_music_url": "",
    "aspect_ratio":    "={{ $('Set Global Config').first().json.aspect_ratio || '9:16' }}",
    "crossfade_secs":  0.5,
    "quality":         "medium"
  }
  Response format: file
  Output property: video
  Timeout: 600000 (10 min — Veo clips can be long)
  continueOnFail: true

  → connects to: ☁️ Upload Rendered Podcast to GCS (same GCS upload node)


═══════════════════════════════════════════════════════════════════════
PART 3 — DYNAMIC REELS (Video Format Agent)
═══════════════════════════════════════════════════════════════════════

This is a SEPARATE output path from the podcast render above.
These nodes produce short-form social Reels/Shorts.
Wire them after Merge: Voice Tracks Ready as a parallel branch
(or after the podcast render, calling /render-dynamic with the same audio).

Node A: 🎨 Video Format Agent
  Type: n8n-nodes-base.openAi
  Model: gpt-4.1-mini
  Credential: OpenAi account
  System prompt: [see n8n-video-render-nodes.json]
  → output[0]: 🖼️ Pixabay: Fetch Images
  → output[0]: 🎵 Pixabay: Fetch Music   (parallel)

Node B: 🖼️ Pixabay: Fetch Images
  GET https://pixabay.com/api/?key={{ $env.PIXABAY_API_KEY }}&q=...&image_type=photo&orientation=vertical&per_page=5&safesearch=true
  → 🔀 Merge: Video Assets (input 1)

Node C: 🎵 Pixabay: Fetch Music
  GET https://pixabay.com/api/?key={{ $env.PIXABAY_API_KEY }}&q=...  (check docs for music endpoint)
  → 🔀 Merge: Video Assets (input 2)

Node D: 🔀 Merge: Video Assets
  mode: combine/mergeByPosition, numberInputs: 2
  → 🎬 Render Dynamic Video

Node E: 🎬 Render Dynamic Video
  POST {{ $env.BACKGROUND_WORKER_URL }}/render-dynamic
  Body: multipart-form-data
  Fields:
    audio_url         = {{ 'https://storage.googleapis.com/' + $('Set Global Config').first().json.gcs_bucket + '/runs/' + $('Set Global Config').first().json.run_id + '/audio/host.mp3' }}
    video_format      = {{ $('🎨 Video Format Agent').first().json.video_format }}
    title             = {{ $('✍️ Voice Writer Agent').first().json.output.episode_title }}
    scripture         = {{ $('🌐 Merge Signal Source').first().json.world_pulse?.calendar?.primary_verse }}
    day_label         = {{ 'Day ' + $('📅 Build Daily Context').first().json.day_of_year + ' · ' + $('📅 Build Daily Context').first().json.day_name }}
    image_urls        = {{ JSON.stringify(($('🖼️ Pixabay: Fetch Images').first().json.hits || []).map(h => h.largeImageURL).filter(Boolean)) }}
    pixabay_music_url = {{ ($('🎵 Pixabay: Fetch Music').first().json.hits || [])[0]?.audio || '' }}
    music_volume      = {{ $('🎨 Video Format Agent').first().json.music_volume }}
    caption_style     = {{ $('🎨 Video Format Agent').first().json.caption_style }}
    viz_style         = {{ $('🎨 Video Format Agent').first().json.viz_style }}
    quality           = medium
  Response format: file → output property: reel_video
  Timeout: 480000
  continueOnFail: true

  → ☁️ Upload Reel to GCS → feed into social publish queue separately


═══════════════════════════════════════════════════════════════════════
SUMMARY — WHAT YOU CAN DISABLE/REMOVE
═══════════════════════════════════════════════════════════════════════

Can DISABLE (Shotstack fully replaced):
  🎞️ Build Assembly Payload
  🎬 Shotstack: Submit Assembly
  ⏳ Wait for Shotstack (60s)
  🔍 Poll Shotstack Status
  ✅ Assembly Ready?
  ⏳ Wait 60s More (Assembly)
  🎞️ Build Short Cut Payload
  🎬 Shotstack: Submit Short Cut
  (+ any Short Cut polling nodes)

Keep ENABLED (still needed):
  🔀 Assembly Mode Router1       (add third branch)
  🧧 Extract Audio URL1          (audio_only fallback)
  🎥 Extract Final Video URL     (add background_worker branch)
  📝 Platform Caption Formatter  (unchanged)
  All publishing nodes           (unchanged)


═══════════════════════════════════════════════════════════════════════
ENV VARS NEEDED
═══════════════════════════════════════════════════════════════════════
  BACKGROUND_WORKER_URL  — https://your-worker.hetzner.io
  PIXABAY_API_KEY        — free at pixabay.com/api/docs

Worker endpoints summary:
  POST /render-podcast      → bg loop + audio → MP4 (replaces Shotstack podcast)
  POST /render-video-clips  → Veo clips + audio → MP4 (replaces Shotstack video)
  POST /render-dynamic      → LLM-directed Reels/Shorts (new — no Shotstack equiv)
  POST /render-audiogram    → basic waveform audiogram (new)
  POST /youtube-context     → transcript extraction (existing)
