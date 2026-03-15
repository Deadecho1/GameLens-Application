# 🎮 GameLens Application

GameLens is a multi-service system for ingesting gameplay captures, classifying events, and running local analysis workflows.

**🧩 Services**
- 📥 `GameLens-Collector` (`backend/GameLens-Collector`) — Flask + Socket.IO ingestion API, stores data in PostgreSQL via `psycopg`, exposes port `8000`. Docker Compose also brings up `postgres` and `pgadmin`.
- 🤖 `GameLens-Event-Classifier` (`backend/GameLens-Event-Classifier`) — FastAPI service for event classification, uses PyTorch + `timm` for inference and PostgreSQL via `psycopg`, exposes port `7761`.
- 🗄️ `PostgreSQL` — Database container used by the backend services.
- 🧰 `pgAdmin` — DB admin UI container.
- 🖥️ `Desktop GUI` (`gui`) — PySide6/Qt desktop client.

**🧱 Stacks**
- 🐍 Python (root project requires `>=3.11,<3.13`; backend services require `>=3.13`).
- ⚙️ FastAPI, Flask, Flask-SocketIO, Gunicorn/eventlet.
- 🧠 PyTorch, `timm`, OpenCV, PaddleOCR, Ultralytics, Transformers.
- 🗄️ PostgreSQL + `psycopg`.
- 🐳 Docker + Docker Compose.
- 🪟 PySide6 (Qt) for the desktop GUI.

**🚀 Run (Docker Compose)**
-  Migrate the DB using the following command, after the startup of all docker containers: 

```bash
cd backend/GameLens-Collector && 
docker exec -i postgres_db psql -U your_username -d your_database_name < db/GameLens-Schema-Updated.sql
 ```
from the main project's root(GameLens-Application), run:

`docker compose up -d`

**📚 Existing Documentation**

Running the (test) Pipeline:
python main.py \
  --video-dir ./data/videos \
  --event-json-dir ./data/event_jsons \
  --run-json-dir ./data/run_jsons

  Run only event detection:
  python main.py \
  --video-dir ./data/videos \
  --event-json-dir ./data/event_jsons \
  --run-json-dir ./data/run_jsons \
  --only-events

  Run only run export:
  python main.py \
  --video-dir ./data/videos \
  --event-json-dir ./data/event_jsons \
  --run-json-dir ./data/run_jsons \
  --only-export
  
-Videos must be .mp4
-All videos should be placed in the specified video folder

Model Weights are available here:
https://drive.google.com/drive/folders/1P8V-G7gfTAqPlpaS92RGeDA0vaGRivSH?usp=sharing
