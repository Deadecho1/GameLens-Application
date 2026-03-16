import json
from pathlib import Path
from typing import Dict, List

from .models import RefinedEvent, RefinedRun


class JsonSerializer:
    @staticmethod
    def event_to_dict(ev: RefinedEvent) -> Dict:
        data: Dict = {
            "time": ev.refined_time,
            "frame": ev.refined_frame,
            "confidence": ev.refined_score,
        }

        if ev.retry_frames:
            data["retry_frames"] = list(ev.retry_frames)
        if ev.retry_times:
            data["retry_times"] = list(ev.retry_times)
        if ev.refinement_method:
            data["refinement_method"] = ev.refinement_method

        return data

    def decoded_runs_to_dict(
        self,
        video_path: str,
        fps: float,
        duration: float,
        refined_runs: List[RefinedRun],
    ) -> Dict:
        runs_json = []

        for i, run in enumerate(refined_runs, start=1):
            runs_json.append(
                {
                    "run_index": i,
                    "start_event": self.event_to_dict(run.start),
                    "end_event": self.event_to_dict(run.end),
                    "choice_events": [self.event_to_dict(ev) for ev in run.choices],
                    "drop_events": [self.event_to_dict(ev) for ev in run.drops],
                }
            )

        return {
            "video_name": Path(video_path).name,
            "fps": fps,
            "duration_seconds": duration,
            "runs": runs_json,
        }

    def save_runs_json(
        self,
        video_path: str,
        fps: float,
        duration: float,
        refined_runs: List[RefinedRun],
        json_out_path: str,
    ) -> None:
        data = self.decoded_runs_to_dict(
            video_path=video_path,
            fps=fps,
            duration=duration,
            refined_runs=refined_runs,
        )

        out_path = Path(json_out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
