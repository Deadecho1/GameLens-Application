import gc
import json
import os
import time
from pathlib import Path
from typing import Optional

from app_core.logging import get_logger

from .choice_service import ChoiceExtractionService
from .json_reader import EventJsonReader
from .models import RunEventJson, VideoEventsJson
from .video_frame_provider import VideoFrameProvider

logger = get_logger(__name__)

_GAMELENS_IMG_SENTINEL = "__GAMELENS_IMG__"

# Persistent debug log written to disk before each risky operation.
# Survives WSL crashes — check this file after a crash to see last known state.
_DEBUG_LOG = Path.home() / "alpha" / "crash_debug.log"
_DEBUG_CHECKPOINT = Path.home() / "alpha" / "crash_checkpoint.txt"


def _get_mem_rss_mb() -> float:
    """Read RSS memory of this process from /proc/self/status (no deps)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def _get_open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1


def _get_tcp_connections() -> int:
    """Count rows in /proc/net/tcp + /proc/net/tcp6 (each row = one connection)."""
    total = 0
    for p in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(p) as f:
                # Subtract 1 for header line
                total += max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    return total


def _debug_write(tag: str, extra: str = "") -> None:
    """Append one line to crash_debug.log and overwrite crash_checkpoint.txt."""
    ts = time.strftime("%H:%M:%S")
    rss = _get_mem_rss_mb()
    fds = _get_open_fds()
    tcp = _get_tcp_connections()
    line = f"[{ts}] {tag} | rss={rss:.0f}MB fds={fds} tcp={tcp}"
    if extra:
        line += f" | {extra}"
    line += "\n"
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        with open(_DEBUG_CHECKPOINT, "w") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass  # never let debug writes break the pipeline


def _maybe_show_frame(image_bytes: bytes, label: str) -> None:
    import base64
    import io
    import os
    import sys
    import tempfile

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((640, 360))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    small_bytes = buf.getvalue()
    b64 = base64.b64encode(small_bytes).decode()

    if not sys.stdout.isatty():
        # Running as a subprocess (e.g. from GUI) — emit sentinel on stdout
        print(f"{_GAMELENS_IMG_SENTINEL}:{b64}", flush=True)
    elif (
        os.environ.get("WT_SESSION")
        or os.environ.get("TERM_PROGRAM") == "iTerm.app"
        or os.environ.get("KITTY_WINDOW_ID")
    ):
        # Terminal with inline image support — OSC 1337
        sys.stderr.write(f"\x1b]1337;File=inline=1:{b64}\a\n")
        sys.stderr.flush()
    else:
        # Plain terminal — save to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=label, delete=False
        ) as tmp:
            tmp.write(image_bytes)
            logger.debug("    -> screenshot: %s", tmp.name)


class RunExporter:

    def __init__(
        self,
        json_reader: EventJsonReader,
        frame_provider: VideoFrameProvider,
        choice_service: ChoiceExtractionService,
    ):
        self.json_reader = json_reader
        self.frame_provider = frame_provider
        self.choice_service = choice_service
        # Phase 1 — scan backward from the detected frame to find the selection moment
        # (the last frame where the choice screen is still visible).
        self.CHOICE_SCAN_STRIDE = 10          # frames between scan steps
        self.CHOICE_MAX_SCAN_LOOKBACK = 300   # max frames to scan back (~5s at 60fps)
        # Phase 2 — once the boundary frame is found, sample this many frames
        # backward from it for the final multi-frame extraction call.
        self.CHOICE_EXTRACTION_OFFSETS = [5]

    def _compute_duration(
        self,
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> Optional[float]:
        if start_time is None or end_time is None:
            return None
        return end_time - start_time

    def _export_single_run(
        self,
        video_name: str,
        run: RunEventJson,
    ) -> dict:
        start_time = run.start_event.time
        end_time = run.end_event.time
        duration = self._compute_duration(start_time, end_time)

        choices = []

        _debug_write(
            f"RUN_START {video_name}",
            f"run={run.run_index} choices={len(run.choice_events)}",
        )
        logger.debug(
            "[%s] run %d: start=%.2fs end=%.2fs, %d choice event(s)",
            video_name, run.run_index, start_time, end_time, len(run.choice_events),
        )

        for ci, choice_event in enumerate(run.choice_events):
            if choice_event.frame is None:
                logger.debug("  choice %d: no frame, skipping", ci)
                continue

            # Phase 1 — scan backward from the detected frame to find the boundary:
            # the last frame where the choice screen is still visible.
            # The event detector fires late (usually after the screen closes), so we
            # walk backward until we see options, then stop.
            boundary_frame: int | None = None
            for offset in range(
                0, self.CHOICE_MAX_SCAN_LOOKBACK + 1, self.CHOICE_SCAN_STRIDE
            ):
                fi = choice_event.frame - offset
                if fi < 0:
                    break
                _debug_write(
                    f"SCAN {video_name}",
                    f"run={run.run_index} choice={ci} offset={offset} frame={fi}",
                )
                fb = self.frame_provider.get_frame_bytes(video_name=video_name, frame_index=fi)
                scan_result = self.choice_service.extract_choice([fb])
                del fb
                gc.collect()
                _debug_write(
                    f"SCAN_DONE {video_name}",
                    f"run={run.run_index} choice={ci} frame={fi} options={scan_result.get('options')}",
                )
                if scan_result.get("options"):
                    boundary_frame = fi
                    logger.debug(
                        "  choice %d: boundary found at frame %d (offset %d from detected)",
                        ci, fi, offset,
                    )
                    break

            if boundary_frame is None:
                logger.debug(
                    "  choice %d: no choice screen found in %d-frame scan, discarding",
                    ci, self.CHOICE_MAX_SCAN_LOOKBACK,
                )
                self.frame_provider.release_video(video_name)
                gc.collect()
                continue

            # Phase 2 — collect frames from the boundary window and send together.
            # Sampling backward from the boundary gives frames where the player's
            # cursor was hovering before the selection was made.
            extraction_frames: list[bytes] = []
            extraction_indices: list[int] = []
            for off in self.CHOICE_EXTRACTION_OFFSETS:
                fi = boundary_frame - off
                if fi >= 0:
                    extraction_frames.append(
                        self.frame_provider.get_frame_bytes(video_name=video_name, frame_index=fi)
                    )
                    extraction_indices.append(fi)

            logger.debug(
                "  choice %d: extracting from frames %s", ci, extraction_indices,
            )
            _debug_write(
                f"EXTRACT {video_name}",
                f"run={run.run_index} choice={ci} boundary={boundary_frame} frames={extraction_indices}",
            )

            result = self.choice_service.extract_choice(extraction_frames)
            for fb in extraction_frames:
                del fb
            del extraction_frames
            gc.collect()

            _debug_write(
                f"EXTRACT_DONE {video_name}",
                f"run={run.run_index} choice={ci} frames={extraction_indices} options={result.get('options')} sel={result.get('selected_option')!r}",
            )
            logger.debug(
                "    -> options=%s selected=%r",
                result.get("options"), result.get("selected_option"),
            )

            if result.get("options"):
                choices.append(result)
            else:
                logger.debug("  choice %d: extraction returned no options, discarding", ci)

            # Release VideoReader after each choice so decord's C++ heap pages
            # are returned to the OS before the next choice begins.
            self.frame_provider.release_video(video_name)
            gc.collect()
            _debug_write(f"RELEASED_AFTER_CHOICE {video_name}", f"run={run.run_index} choice={ci}")

        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "choices": choices,
        }

    def _make_output_filename(self, video_name: str, run_index: int) -> str:
        stem = Path(video_name).stem
        return f"{stem}_run_{run_index}.json"

    def process_video_json(
        self,
        json_path: Path,
        output_dir: Path,
        verbose: bool = False,
    ) -> None:
        video_events: VideoEventsJson = self.json_reader.read_video_events(json_path)
        logger.info("Processing %s: %d run(s)", video_events.video_name, len(video_events.runs))

        try:
            for run in video_events.runs:
                logger.info("  [run %d/%d] exporting...", run.run_index, len(video_events.runs))
                exported = self._export_single_run(
                    video_name=video_events.video_name,
                    run=run,
                )

                out_name = self._make_output_filename(
                    video_name=video_events.video_name,
                    run_index=run.run_index,
                )
                out_path = output_dir / out_name

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(exported, f, indent=2, ensure_ascii=False)

                logger.info("  [run %d] saved %d choice(s) -> %s", run.run_index, len(exported["choices"]), out_path.name)

                # Release VideoReader after each run so decord's internal decode
                # buffers are freed before the next run's frame seeks begin.
                self.frame_provider.release_video(video_events.video_name)
                gc.collect()
                _debug_write(
                    f"RUN_DONE {video_events.video_name}",
                    f"run={run.run_index} choices_saved={len(exported['choices'])}",
                )
        finally:
            # Ensure release even if a run raised an exception.
            self.frame_provider.release_video(video_events.video_name)

    def process_folder(
        self,
        json_dir: Path,
        output_dir: Path,
        verbose: bool = False,
    ) -> None:
        json_files = self.json_reader.list_json_files(json_dir)

        if not json_files:
            logger.warning("No JSON files found in: %s", json_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        _debug_write("=== PROCESS_FOLDER START ===", f"videos={len(json_files)}")

        ok = 0
        failed = 0

        for idx, json_path in enumerate(json_files, start=1):
            logger.info("[%d/%d] Processing: %s", idx, len(json_files), json_path.name)
            _debug_write(f"VIDEO_START {json_path.stem}", f"idx={idx}/{len(json_files)}")

            try:
                self.process_video_json(
                    json_path=json_path,
                    output_dir=output_dir,
                    verbose=verbose,
                )
                ok += 1
            except Exception as e:
                logger.error("FAILED: %s — %s", json_path.name, e)
                failed += 1
            finally:
                # Reset HTTP session between videos to release connection pool
                # resources accumulated during that video's API calls.
                self.choice_service.choice_extractor.reset_session()
                gc.collect()

        logger.info("Done. Successful: %d  Failed: %d", ok, failed)