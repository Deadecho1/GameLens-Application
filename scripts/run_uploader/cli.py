from __future__ import annotations

import argparse
from pathlib import Path

from .uploader import RunUploader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload completed run JSONs to the GameLens Collector service."
    )
    parser.add_argument("--run-json-dir", required=True, help="Directory containing *_run_*.json files")
    parser.add_argument("--game-name", required=True, help="Game name to associate runs with")
    parser.add_argument("--version-name", required=True, help="Game version name")
    parser.add_argument("--user-id", default="1", help="User ID sent in X-User-ID header")
    parser.add_argument("--collector-url", default="http://localhost:8000", help="Collector base URL")
    args = parser.parse_args()

    uploader = RunUploader(collector_url=args.collector_url, user_id=args.user_id)
    uploader.upload_from_dir(
        run_json_dir=Path(args.run_json_dir),
        game_name=args.game_name,
        version_name=args.version_name,
    )


if __name__ == "__main__":
    main()
