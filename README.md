# GameLens

GameLens is a gameplay analysis system that processes video captures to detect in-game events, classify game-state transitions, and export structured run/session data. Results are stored locally (SQLite) and optionally synced to a remote Collector when signed in.

---

## Architecture

```
Videos → Event Detection → Choice Extraction → Run JSON files
                                                      ↓
                                              Local SQLite DB
                                                      ↓ (on login)
                                              Remote Collector (PostgreSQL)
                                                      ↓
                                          Electron + React analytics UI
```

**Components:**

| Component | Location | Description |
|-----------|----------|-------------|
| Electron + React UI | `frontend/src/` | Main user interface |
| Qt IPC backend | `frontend/gui/` | PySide6 headless backend; bridges Electron ↔ pipeline |
| GameLens-Collector | `backend/GameLens-Collector/` | Flask + Socket.IO ingestion API, port `8000` |
| GameLens-Event-Extraction | `backend/GameLens-Event-Extraction/` | FastAPI ML classification service, port `7761` |
| PostgreSQL | Docker | Remote database (Collector) |
| Local SQLite | `data/gamelens_local.db` | Per-machine local storage, no login required |

---

## Prerequisites

- **Python 3.11–3.12** (root project and GUI)
- **Node.js 18+** and **npm**
- **[uv](https://github.com/astral-sh/uv)** — Python package manager
- **Docker + Docker Compose** — for remote backend services (optional for local-only use)
- **CUDA-capable GPU** — optional, strongly recommended for event detection

---

## 1. Install Dependencies

```bash
# Python
uv sync

# Node (first time only)
cd frontend && npm install && cd ..
```

---

## 2. Download Model Weights

Weights are not in the repo. Download from Google Drive and place as shown:

**Google Drive:** https://drive.google.com/drive/folders/1P8V-G7gfTAqPlpaS92RGeDA0vaGRivSH?usp=sharing

```
models/
├── event_detector/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── boss/
    └── model.pt
```

---

## 3. Configure Environment

### Root `.env`

```env
OPENAI_API_KEY=your_openai_api_key
```

### `backend/GameLens-Collector/.env`

```bash
cp backend/GameLens-Collector/.env.example backend/GameLens-Collector/.env
```

```env
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database_name

# Docker: @db:5432 | Local dev: @localhost:5432
PGSQL_CONN=postgresql://your_username:your_password@db:5432/your_database_name

CLASSIFIER_SERVICE_HOST_URL=http://event_classifier:7761

# Optional
PGADMIN_MAIL=admin@gamelens.com
PGADMIN_PASS=admin123
```

### `backend/GameLens-Event-Extraction/.env`

```bash
cp backend/GameLens-Event-Extraction/.env.example backend/GameLens-Event-Extraction/.env
```

```env
PGSQL_CONN=postgresql://your_username:your_password@db:5432/your_database_name
OPENAI_API_KEY=your_openai_api_key
```

---

## 4. Launch the App

```bash
cd frontend && npm run electron:dev
```

**This is all you need for local-only use.** Docker is not required to process videos or view analytics. Data is saved to `data/gamelens_local.db`.

### Mock mode (frontend dev, no Qt backend)

```bash
cd frontend && npm run electron:dev:mock
```

Returns pre-shaped fixture data. No Qt process needed.

---

## 5. Backend Services (for remote sync)

Start Docker services:

```bash
docker network create db_network   # once
docker compose up -d --build
```

Apply DB schemas (once, or after wiping the volume):

```bash
docker exec -i postgres_db psql -U your_username -d your_database_name \
  < backend/GameLens-Collector/db/GameLens-Schema-Updated.sql

docker exec -i postgres_db psql -U your_username -d your_database_name \
  < backend/GameLens-Collector/db/dashboard_schema.sql

docker exec -i postgres_db psql -U your_username -d your_database_name \
  < backend/GameLens-Collector/db/migrations/001_add_runs_dedup.sql
```

Verify:

```bash
docker compose ps
curl http://localhost:8000   # Collector
curl http://localhost:7761   # Event Extraction
```

---

## 6. Sign In and Sync

Click **Sign in** in the app header. Enter any email — an account is created automatically. After login:

- Analytics reads from the remote Collector
- All locally saved runs sync to the remote DB in the background
- Data becomes accessible from any machine signed in with the same email

Sign out to return to local-only mode.

> Password auth is not yet implemented. Any email string is accepted.

---

## CLI Pipeline

Runs the three processing stages manually. Stage 1 only needs Python. Stages 2–3 need the backend services running.

### Stage 1 — Event Detection

```bash
uv run python -m scripts.event_detector.cli \
  --input-dir /path/to/videos \
  --output-dir /path/to/event-jsons
```

### Stage 2 — Run Exporter

```bash
uv run python -m scripts.run_exporter.cli \
  --json-dir /path/to/event-jsons \
  --video-dir /path/to/videos \
  --output-dir /path/to/run-jsons
```

Stage 2 output is safe to reuse — re-running is expensive (one OpenAI call per choice event).

### Stage 3 — Save to Local DB

```bash
uv run python -m scripts.run_uploader.cli \
  --run-json-dir /path/to/run-jsons \
  --game-name "MyGame" \
  --version-name "v1.0" \
  --backend local
```

To upload directly to the remote Collector instead:

```bash
uv run python -m scripts.run_uploader.cli \
  --run-json-dir /path/to/run-jsons \
  --game-name "MyGame" \
  --version-name "v1.0" \
  --backend remote \
  --user-id 1 \
  --collector-url http://localhost:8000
```

The GUI runs all three stages automatically when you click **Run** in the Processing tab.

---

## Notes

- Videos must be `.mp4`
- Event detector uses X-CLIP; runs on GPU if CUDA is available, otherwise CPU (slow)
- Choice extraction requires `OPENAI_API_KEY`
- Local SQLite is at `data/gamelens_local.db` — safe to delete to reset local data
- Fine-tuned models go in `models/finetuned/<name>_<modelId>/`
