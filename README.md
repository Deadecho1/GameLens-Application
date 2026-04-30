# GameLens

GameLens is a gameplay analysis system that processes video captures to detect in-game events, classify game-state transitions, and export structured run/session data.

**Services**
- `GameLens-Collector` (`backend/GameLens-Collector`) — Flask + Socket.IO ingestion API. Stores data in PostgreSQL. Port `8000`.
- `GameLens-Event-Extraction` (`backend/GameLens-Event-Extraction`) — FastAPI service for event/choice classification using vision models + OpenAI. Port `7761`.
- `PostgreSQL` — Database used by both backend services.
- `pgAdmin` — Optional DB admin UI. Port `5050`.
- `GUI` (`gui/`) — PySide6 desktop client that connects to the backend.

---

## Prerequisites

- **Python 3.11–3.12** (root project and GUI)
- **[uv](https://github.com/astral-sh/uv)** — Python package manager
- **Docker + Docker Compose** — for running backend services
- **CUDA-capable GPU** (optional but strongly recommended for event detection inference)

---

## 1. Install Python Dependencies

From the repo root:

```bash
uv sync
```

---

## 2. Download Model Weights

Model weights are not included in the repository. Download them from Google Drive and place them as follows:

**Google Drive:** https://drive.google.com/drive/folders/1P8V-G7gfTAqPlpaS92RGeDA0vaGRivSH?usp=sharing

Download the event detector weights and place them at `models/event_detector/`:

```
models/
└── event_detector/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ... (other tokenizer/preprocessor files)
```

---

## 3. Configure Environment Files

### `backend/GameLens-Collector/.env`

Copy the example and fill in your values:

```bash
cp backend/GameLens-Collector/.env.example backend/GameLens-Collector/.env
```

```env
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database_name

# Docker: use @db:5432 | Local: use @localhost:5432
PGSQL_CONN=postgresql://your_username:your_password@db:5432/your_database_name

# URL of the Event Extraction service (use container name when running via Docker)
CLASSIFIER_SERVICE_HOST_URL=http://event_classifier:7761

# Optional — pgAdmin credentials
PGADMIN_MAIL=admin@gamelens.com
PGADMIN_PASS=admin123
```

### `backend/GameLens-Event-Extraction/.env`

```bash
cp backend/GameLens-Event-Extraction/.env.example backend/GameLens-Event-Extraction/.env
```

```env
# Docker: use @db:5432 | Local: use @localhost:5432
PGSQL_CONN=postgresql://your_username:your_password@db:5432/your_database_name

OPENAI_API_KEY=your_openai_api_key
```

### Root `.env`

Create a `.env` in the repo root for the GUI and CLI pipeline:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## 4. Create the Docker Network

Both backend services share an external Docker network. Create it once before the first run:

```bash
docker network create db_network
```

---

## 5. Start Backend Services

From the repo root:

```bash
docker compose up -d --build
```

This starts: `postgres_db`, `pgadmin`, `collector` (port 8000), and `event_classifier` (port 7761).

Wait for all containers to be healthy, then run the DB migration:

```bash
docker exec -i postgres_db psql -U your_username -d your_database_name < backend/GameLens-Collector/db/GameLens-Schema-Updated.sql
```

> The migration only needs to be run once (or after wiping the DB volume).

Verify services are up:

```bash
docker compose ps
curl http://localhost:8000      # Collector
curl http://localhost:7761      # Event Extraction
```

---

## 6. Launch the GUI

```bash
cd frontend
npm install        # first time only
npm run electron:dev
```

The GUI connects to:
- Collector API at `http://localhost:8000`
- Event Extraction service at `http://localhost:7761` (configurable via `GAMELENS_CLASSIFIER_URL` in `.env`)

---

---

## CLI Pipeline

The CLI pipeline processes videos in three sequential stages. The backend must be running (`docker compose up -d --build`) before Stage 2 and Stage 3.

### Stage 1 — Event Detection

Processes video files and outputs per-video JSON files describing detected run boundaries, choice events, and drop events.

```bash
uv run python -m scripts.event_detector.cli \
  --input-dir /path/to/videos \
  --output-dir /path/to/event-jsons
```

### Stage 2 — Run Exporter

Reads the event JSONs from Stage 1, extracts choice selections via the Event Extraction service (OpenAI vision), and writes one JSON file per run.

```bash
uv run python -m scripts.run_exporter.cli \
  --json-dir /path/to/event-jsons \
  --video-dir /path/to/videos \
  --output-dir /path/to/run-jsons
```

Stage 2 output is safe to keep and reuse — re-running it is expensive (one OpenAI call per choice event).

### Stage 3 — Boss Processor

Enriches the run JSONs from Stage 2 with boss fight data. Requires a YOLO boss classifier model and the Event Extraction service.

```bash
uv run python -m scripts.boss_processor.cli \
  --run-json-dir /path/to/run-jsons \
  --video-dir /path/to/videos \
  --boss-model models/boss/model.pt
```

Each run JSON gains a `boss_fights` key:

```json
"boss_fights": [
  {
    "boss_names": ["Stone Golem", "Stone Golem"],
    "boss_class": "boss",
    "start_time": 238.75,
    "end_time": 261.25,
    "duration_seconds": 22.5,
    "player_died": false
  }
]
```

- `boss_names` — all boss names visible on screen (there can be multiple simultaneous bosses).
- `player_died` — `true` if the run ended at or near the boss fight's end time.
- Boss model weights go in `models/boss/` (download separately, same Google Drive as event detector weights).

---

## Notes

- Videos must be `.mp4` format.
- All videos should be placed in the folder configured in the GUI.
- The event detector uses X-CLIP and runs on GPU if CUDA is available, otherwise falls back to CPU (significantly slower).
- Choice extraction uses the OpenAI vision API — an `OPENAI_API_KEY` is required.
