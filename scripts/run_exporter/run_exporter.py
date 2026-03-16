import json
from pathlib import Path
from typing import Optional

from app_core.logging import get_logger

from .choice_service import ChoiceExtractionService
from .json_reader import EventJsonReader
from .models import RunEventJson, VideoEventsJson
from .video_frame_provider import VideoFrameProvider

logger = get_logger(__name__)


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
        self.CHOICE_LOOKBACK_FRAMES = 60
        self.CHOICE_LOOKBACK_STRIDE = 10

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

        logger.debug(
            "[%s] run %d: start=%.2fs end=%.2fs, %d choice event(s)",
            video_name, run.run_index, start_time, end_time, len(run.choice_events),
        )

        for ci, choice_event in enumerate(run.choice_events):
            if choice_event.frame is None:
                logger.debug("  choice %d: no frame, skipping", ci)
                continue

            logger.debug("  choice %d: frame=%d, trying up to %d offsets", ci, choice_event.frame, self.CHOICE_LOOKBACK_FRAMES // self.CHOICE_LOOKBACK_STRIDE + 1)
            choice_result = None

            # Try detected frame, then stride backwards
            for offset in range(0, self.CHOICE_LOOKBACK_FRAMES + 1, self.CHOICE_LOOKBACK_STRIDE):
                frame_index = choice_event.frame - offset
                if frame_index < 0:
                    break

                logger.debug("    offset=%d frame_index=%d — calling extract_choice", offset, frame_index)
                frame_bytes = self.frame_provider.get_frame_bytes(
                    video_name=video_name,
                    frame_index=frame_index,
                )

                result = self.choice_service.extract_choice(frame_bytes)
                logger.debug("    -> options=%s", result.get("options"))

                if result["options"]:
                    choice_result = result
                    logger.debug("    -> found choices at offset=%d", offset)
                    break

            # If still empty after retries → discard event as noise
            if choice_result is None:
                logger.debug("  choice %d: no options found after all offsets, discarding", ci)
                continue

            choices.append(choice_result)

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
        finally:
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

        ok = 0
        failed = 0

        for idx, json_path in enumerate(json_files, start=1):
            logger.info("[%d/%d] Processing: %s", idx, len(json_files), json_path.name)

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

        logger.info("Done. Successful: %d  Failed: %d", ok, failed)