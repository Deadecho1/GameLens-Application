import json
from pathlib import Path
from typing import Optional

from .choice_service import ChoiceExtractionService
from .json_reader import EventJsonReader
from .models import RunEventJson, VideoEventsJson
from .video_frame_provider import VideoFrameProvider


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
        self.CHOICE_LOOKBACK_FRAMES = 16

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

        for choice_event in run.choice_events:
            if choice_event.frame is None:
                continue

            choice_result = None

            # Try detected frame, then a few frames before it
            for offset in range(self.CHOICE_LOOKBACK_FRAMES + 1):
                frame_index = choice_event.frame - offset
                if frame_index < 0:
                    break

                frame_bytes = self.frame_provider.get_frame_bytes(
                    video_name=video_name,
                    frame_index=frame_index,
                )

                result = self.choice_service.extract_choice(frame_bytes)

                if result["options"]:
                    choice_result = result
                    break

            # If still empty after retries → discard event as noise
            if choice_result is None:
                continue

            choices.append(choice_result)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
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

        try:
            for run in video_events.runs:
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

                if verbose:
                    print(f"Saved: {out_path}")
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
            print(f"No JSON files found in: {json_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        failed = 0

        for idx, json_path in enumerate(json_files, start=1):
            print(f"[{idx}/{len(json_files)}] Processing: {json_path.name}")

            try:
                self.process_video_json(
                    json_path=json_path,
                    output_dir=output_dir,
                    verbose=verbose,
                )
                ok += 1
            except Exception as e:
                print(f"FAILED: {json_path.name}")
                print(f"Reason: {e}\n")
                failed += 1

        print("Done.")
        print(f"Successful: {ok}")
        print(f"Failed: {failed}")