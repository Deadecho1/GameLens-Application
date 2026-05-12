from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal, Slot

from .config import PROJECT_ROOT
from .tuning_config import TUNING_MODEL_CONFIG_BY_ID


class TuningRunner(QObject):
    log_message = Signal(str)
    tuning_finished = Signal(bool, str)
    busy_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._process: Optional[QProcess] = None
        self._queue: list[tuple[str, list[str]]] = []
        self._stopping = False
        self._output_buffer: str = ""
        self._status: str = "idle"
        self._temp_dir: Optional[str] = None

    @property
    def status(self) -> str:
        return self._status

    def is_running(self) -> bool:
        return self._process is not None

    @Slot(object)
    def start_tuning(
        self,
        videos: list[dict],
        enabled_model_ids: list[str],
        model_name: str,
        finetuned_models_dir: Path,
        base_model_dir: Path,
        boss_model_path: Optional[Path] = None,
    ) -> None:
        if self._process is not None:
            self.log_message.emit("[WARN] A tuning run is already in progress.\n")
            return

        self._stopping = False
        self._status = "running"
        self._queue = []

        temp_dir = tempfile.mkdtemp(prefix="gamelens_tuning_")
        self._temp_dir = temp_dir

        for model_id in enabled_model_ids:
            config = TUNING_MODEL_CONFIG_BY_ID.get(model_id)
            if config is None:
                self.log_message.emit(f"[WARN] Unknown model id '{model_id}', skipping.\n")
                continue

            clips_dir = Path(temp_dir) / model_id
            clips_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(finetuned_models_dir / f"{model_name}_{model_id}")

            if config.annotation_type == "segment":
                self._queue_segment_model(
                    model_id=model_id,
                    config=config,
                    videos=videos,
                    clips_dir=clips_dir,
                    output_dir=output_dir,
                    boss_model_path=boss_model_path,
                    temp_dir=temp_dir,
                )
            else:
                self._queue_point_model(
                    model_id=model_id,
                    config=config,
                    videos=videos,
                    clips_dir=clips_dir,
                    output_dir=output_dir,
                    base_model_dir=base_model_dir,
                    temp_dir=temp_dir,
                )

        if not self._queue:
            self._status = "idle"
            self.pipeline_finished_helper(False, "Nothing to run — no valid annotations found.")
            return

        self.busy_changed.emit(True)
        self.log_message.emit("=== GameLens Fine-Tuning Started ===\n")
        self._start_next_command()

    def _queue_point_model(
        self,
        model_id: str,
        config,
        videos: list[dict],
        clips_dir: Path,
        output_dir: str,
        base_model_dir: Path,
        temp_dir: str,
    ) -> None:
        all_model_marks: list[dict] = []
        for video in videos:
            for mark in video.get("marks", []):
                if mark.get("modelId") == model_id and mark.get("type", "point") == "point":
                    all_model_marks.append(mark)

        if not all_model_marks:
            self.log_message.emit(f"[WARN] No marks for model '{model_id}', skipping.\n")
            return

        cat_counts: dict[str, int] = {}
        for mark in all_model_marks:
            cat = mark.get("categoryId", "")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        max_cat_count = max(cat_counts.values()) if cat_counts else 0
        none_target = min(max_cat_count, 10)

        for video in videos:
            video_path = video.get("path", "")
            duration = float(video.get("duration") or 0)
            video_marks = [
                m for m in video.get("marks", [])
                if m.get("modelId") == model_id and m.get("type", "point") == "point"
            ]
            if not video_marks:
                continue

            marked_times = [float(m["time"]) for m in video_marks]
            video_none_count = 0
            if none_target > 0 and duration > 0:
                total_marks = len(all_model_marks)
                video_none_count = max(1, round(none_target * len(video_marks) / total_marks))

            none_events = _compute_none_events(marked_times, duration, video_none_count)
            events = [
                {"time": m["time"], "label": m["categoryId"]} for m in video_marks
            ] + none_events

            events_json_path = Path(temp_dir) / f"events_{id(video)}.json"
            events_json_path.write_text(json.dumps(events))

            self._queue.append((
                f"Extracting clips from {Path(video_path).name}",
                [
                    sys.executable, "-m", "scripts.clip_creator.cli",
                    "--video", video_path,
                    "--output-dir", str(clips_dir),
                    "--events-json", str(events_json_path),
                    "--duration", "2.0",
                ],
            ))

        self._queue.append((
            f"Fine-tuning {config.label}",
            [
                sys.executable, "-m", config.train_script,
                "--clips-dir", str(clips_dir),
                "--output-dir", output_dir,
                "--base-model-dir", str(base_model_dir),
            ],
        ))

    def _queue_segment_model(
        self,
        model_id: str,
        config,
        videos: list[dict],
        clips_dir: Path,
        output_dir: str,
        boss_model_path: Optional[Path],
        temp_dir: str,
    ) -> None:
        if boss_model_path is None:
            self.log_message.emit(f"[WARN] boss_model_path not set, skipping '{model_id}'.\n")
            return

        segment_videos = [
            {
                "path": v["path"],
                "duration": v.get("duration", 0),
                "marks": [
                    m for m in v.get("marks", [])
                    if m.get("type") == "segment" and m.get("modelId") == model_id
                ],
            }
            for v in videos
            if any(
                m.get("type") == "segment" and m.get("modelId") == model_id
                for m in v.get("marks", [])
            )
        ]

        if not segment_videos:
            self.log_message.emit(f"[WARN] No segments for model '{model_id}', skipping.\n")
            return

        input_json_path = Path(temp_dir) / f"input_{model_id}.json"
        input_json_path.write_text(json.dumps(segment_videos))

        self._queue.append((
            f"Extracting boss frames",
            [
                sys.executable, "-m", "scripts.finetuner.extract_frames_cli",
                "--input-json", str(input_json_path),
                "--output-dir", str(clips_dir),
            ],
        ))

        self._queue.append((
            f"Fine-tuning {config.label}",
            [
                sys.executable, "-m", config.train_script,
                "--clips-dir", str(clips_dir),
                "--output-dir", output_dir,
                "--base-model-path", str(boss_model_path),
            ],
        ))

    def pipeline_finished_helper(self, success: bool, message: str) -> None:
        self.log_message.emit(message + "\n")
        self.tuning_finished.emit(success, message)

    @Slot()
    def stop_tuning(self) -> None:
        if self._process is None:
            return
        self._stopping = True
        self.log_message.emit("Stopping fine-tuning...\n")
        if os.name == "nt":
            self._process.kill()
        else:
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                self._process.kill()

    def _start_next_command(self) -> None:
        if not self._queue:
            self._cleanup_process()
            self._status = "completed"
            self.busy_changed.emit(False)
            self.tuning_finished.emit(True, "Fine-tuning completed successfully.")
            return

        stage_name, command = self._queue.pop(0)
        self.log_message.emit(f"\n=== {stage_name} ===\n")

        process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessEnvironment(self._build_env())
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._handle_finished)
        process.errorOccurred.connect(self._handle_error)

        self._process = process
        process.start(command[0], command[1:])

    def _build_env(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH", "")
        root = str(PROJECT_ROOT)
        env.insert("PYTHONPATH", root + os.pathsep + existing if existing else root)
        return env

    @Slot()
    def _read_output(self) -> None:
        if self._process is None:
            return
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        if not data:
            return
        self._output_buffer += data
        pos = self._output_buffer.rfind("\n")
        if pos != -1:
            self.log_message.emit(self._output_buffer[: pos + 1])
            self._output_buffer = self._output_buffer[pos + 1:]

    def _flush_output_buffer(self) -> None:
        if self._process is not None:
            remaining = self._process.readAllStandardOutput().data().decode(errors="replace")
            if remaining:
                self._output_buffer += remaining
        if self._output_buffer:
            self.log_message.emit(self._output_buffer)
            self._output_buffer = ""

    @Slot(int, QProcess.ExitStatus)
    def _handle_finished(self, exit_code: int, _: QProcess.ExitStatus) -> None:
        if self._process is None:
            return
        self._flush_output_buffer()

        if self._stopping:
            self._cleanup_process()
            self._status = "stopped"
            self.busy_changed.emit(False)
            self.tuning_finished.emit(False, "Fine-tuning stopped by user.")
            return

        if exit_code != 0:
            self._cleanup_process()
            self._status = "stopped"
            self.busy_changed.emit(False)
            self.tuning_finished.emit(False, f"Stage failed (exit code {exit_code}).")
            return

        self.log_message.emit("\n")
        self._cleanup_process()
        self._start_next_command()

    @Slot(QProcess.ProcessError)
    def _handle_error(self, error: QProcess.ProcessError) -> None:
        self._cleanup_process()
        self._status = "stopped"
        self.busy_changed.emit(False)
        self.tuning_finished.emit(False, f"Process error: {error}.")

    def _cleanup_process(self) -> None:
        if self._process is None:
            return
        self._process.deleteLater()
        self._process = None
        self._output_buffer = ""


def _compute_none_events(
    marked_times: list[float],
    duration: float,
    target_count: int,
    min_gap: float = 5.0,
) -> list[dict]:
    if target_count <= 0 or duration <= 0:
        return []
    candidates: list[float] = []
    step = 3.0
    t = 1.0
    while t < duration - 1.0:
        if all(abs(t - mt) >= min_gap for mt in marked_times):
            candidates.append(round(t, 2))
        t += step
    random.shuffle(candidates)
    return [{"time": t, "label": "none"} for t in candidates[:target_count]]
