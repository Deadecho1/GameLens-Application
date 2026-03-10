import argparse
import subprocess
import sys
from pathlib import Path


def validate_input_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a folder: {path}")


def run_command(command: list[str]) -> None:
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def run_event_detector(
    video_dir: Path,
    event_json_dir: Path,
    verbose: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.event_detector.cli",
        "--input-dir",
        str(video_dir),
        "--output-dir",
        str(event_json_dir),
    ]

    if verbose:
        command.append("--verbose")

    run_command(command)


def run_run_exporter(
    event_json_dir: Path,
    video_dir: Path,
    run_json_dir: Path,
    verbose: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_exporter.cli",
        "--json-dir",
        str(event_json_dir),
        "--video-dir",
        str(video_dir),
        "--output-dir",
        str(run_json_dir),
    ]

    if verbose:
        command.append("--verbose")

    run_command(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local GameLens event pipeline on a folder of MP4 videos."
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Folder containing the source .mp4 files",
    )
    parser.add_argument(
        "--event-json-dir",
        required=True,
        help="Folder to write per-video event JSON files",
    )
    parser.add_argument(
        "--run-json-dir",
        required=True,
        help="Folder to write per-run output JSON files",
    )
    parser.add_argument(
        "--only-events",
        action="store_true",
        help="Run only the event detector stage",
    )
    parser.add_argument(
        "--only-export",
        action="store_true",
        help="Run only the run-export stage using existing event JSON files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )

    args = parser.parse_args()

    if args.only_events and args.only_export:
        raise ValueError("Choose only one of --only-events or --only-export")

    video_dir = Path(args.video_dir).resolve()
    event_json_dir = Path(args.event_json_dir).resolve()
    run_json_dir = Path(args.run_json_dir).resolve()

    validate_input_dir(video_dir, "Video directory")
    event_json_dir.mkdir(parents=True, exist_ok=True)
    run_json_dir.mkdir(parents=True, exist_ok=True)

    if not args.only_export:
        print("=== Stage 1: Event detector ===")
        run_event_detector(
            video_dir=video_dir,
            event_json_dir=event_json_dir,
            verbose=args.verbose,
        )
        print()

    if not args.only_events:
        print("=== Stage 2: Run exporter ===")
        run_run_exporter(
            event_json_dir=event_json_dir,
            video_dir=video_dir,
            run_json_dir=run_json_dir,
            verbose=args.verbose,
        )
        print()

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()