# media-worker

Small FastAPI service for n8n: **YouTube URL → captions or Whisper transcript** (mono 16 kHz, `faster-whisper`).

## Run on a VPS (Docker)

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO   # or cd media-worker if this repo is only the worker
docker compose up -d --build
curl -s http://127.0.0.1:8000/health
```

- Worker listens on **8000** inside the stack.
- With the bundled `docker-compose.yml`, n8n can call **`http://media-worker:8000/youtube-context`** on the same Docker network.

## API

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| POST | `/youtube-context` | `{"url": "https://youtube.com/...", "prefer_captions": true}` |

## Environment (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, … |
| `WHISPER_DEVICE` | `cpu` | `cuda` if GPU |
| `WHISPER_COMPUTE_TYPE` | `int8` | e.g. `float16` on GPU |
| `YTDLP_SUB_LANGS` | `en.*,en-US.*` | yt-dlp subtitle languages |

## n8n

Import `n8n-workflow-youtube-context.json` and point the HTTP Request node at your worker URL.

## License

Use and modify for your own stack.
