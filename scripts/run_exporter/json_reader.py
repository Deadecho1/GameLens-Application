import json
from pathlib import Path
from typing import List

from .models import EventPoint, RunEventJson, VideoEventsJson


class EventJsonReader:
    def list_json_files(self, json_dir: Path) -> List[Path]:
        return sorted(
            p for p in json_dir.glob("*.json")
            if p.is_file()
        )

    def _parse_event_point(self, data: dict) -> EventPoint:
        return EventPoint(
            time=data.get("time"),
            frame=data.get("frame"),
            confidence=data.get("confidence"),
        )

    def read_video_events(self, json_path: Path) -> VideoEventsJson:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        runs = []
        for run_data in data.get("runs", []):
            runs.append(
                RunEventJson(
                    run_index=run_data["run_index"],
                    start_event=self._parse_event_point(run_data["start_event"]),
                    end_event=self._parse_event_point(run_data["end_event"]),
                    choice_events=[
                        self._parse_event_point(x)
                        for x in run_data.get("choice_events", [])
                    ],
                    drop_events=[
                        self._parse_event_point(x)
                        for x in run_data.get("drop_events", [])
                    ],
                )
            )

        return VideoEventsJson(
            video_name=data["video_name"],
            fps=data["fps"],
            duration_seconds=data["duration_seconds"],
            runs=runs,
        )