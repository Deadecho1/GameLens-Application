import json
from pathlib import Path
from typing import Dict, List

from .models import DecodedRun, PeakEvent


class JsonSerializer:
    @staticmethod
    def event_to_dict(ev: PeakEvent) -> Dict:
        t = ev.refined_time if ev.refined_time is not None else ev.time
        s = ev.refined_score if ev.refined_score is not None else ev.score
        frame = ev.refined_frame if ev.refined_frame is not None else None

        return {
            "time": t,
            "frame": frame,
            "confidence": s,
        }

    def decoded_runs_to_dict(
        self,
        video_path: str,
        fps: float,
        duration: float,
        decoded_runs: List[DecodedRun],
    ) -> Dict:
        runs_json = []

        for i, run in enumerate(decoded_runs, start=1):
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
        decoded_runs: List[DecodedRun],
        json_out_path: str,
    ) -> None:
        data = self.decoded_runs_to_dict(
            video_path=video_path,
            fps=fps,
            duration=duration,
            decoded_runs=decoded_runs,
        )

        out_path = Path(json_out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)