from __future__ import annotations

import argparse
from pathlib import Path

from .local_uploader import LocalRunUploader
from .uploader import RunUploader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload completed run JSONs to local DB or remote Collector."
    )
    parser.add_argument("--run-json-dir", required=True, help="Directory containing *_run_*.json files")
    parser.add_argument("--game-name", required=True, help="Game name")
    parser.add_argument("--version-name", required=True, help="Game version name")
    parser.add_argument(
        "--backend",
        choices=["local", "remote"],
        default="local",
        help="local: write to local SQLite (default); remote: POST to Collector",
    )
    parser.add_argument("--user-id", default="1", help="User ID (remote mode only)")
    parser.add_argument("--collector-url", default="http://localhost:8000", help="Collector base URL (remote mode only)")
    parser.add_argument("--db-path", default=None, help="SQLite path (local mode only, defaults to data/gamelens_local.db)")
    args = parser.parse_args()

    run_json_dir = Path(args.run_json_dir)

    if args.backend == "local":
        db_path = Path(args.db_path) if args.db_path else None
        LocalRunUploader(db_path=db_path).upload_from_dir(
            run_json_dir=run_json_dir,
            game_name=args.game_name,
            version_name=args.version_name,
        )
    else:
        RunUploader(collector_url=args.collector_url, user_id=args.user_id).upload_from_dir(
            run_json_dir=run_json_dir,
            game_name=args.game_name,
            version_name=args.version_name,
        )


if __name__ == "__main__":
    main()
