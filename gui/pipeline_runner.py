from __future__ import annotations

import os
import signal
import sys
from typing import Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal, Slot

from .config import PROJECT_ROOT
from .models import PipelineConfig


class PipelineRunner(QObject):
    log_message = Signal(str)
    stage_changed = Signal(str)
    pipeline_finished = Signal(bool, str)
    busy_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._process: Optional[QProcess] = None
        self._config: Optional[PipelineConfig] = None
        self._queue: list[tuple[str, list[str]]] = []
        self._stopping = False
        self._output_buffer: str = ""

    def is_running(self) -> bool:
        return self._process is not None

    @Slot(object)
    def start_pipeline(self, config: PipelineConfig) -> None:
        if self._process is not None:
            self.log_message.emit("A pipeline is already running.")
            return

        self._config = config
        self._stopping = False
        self._queue = []

        if not config.only_export:
            self._queue.append(
                (
                    "Detecting Events...",
                    [
                        sys.executable,
                        "-m",
                        "scripts.event_detector.cli",
                        "--input-dir",
                        str(config.video_dir),
                        "--output-dir",
                        str(config.event_json_dir),
                    ]
                    + (["--verbose"] if config.verbose else []),
                )
            )

        if not config.only_events:
            self._queue.append(
                (
                    "Processing Events...",
                    [
                        sys.executable,
                        "-m",
                        "scripts.run_exporter.cli",
                        "--json-dir",
                        str(config.event_json_dir),
                        "--video-dir",
                        str(config.video_dir),
                        "--output-dir",
                        str(config.run_json_dir),
                    ]
                    + (["--verbose"] if config.verbose else []),
                )
            )

        if not self._queue:
            self.pipeline_finished.emit(False, "Nothing to run.")
            return

        self.busy_changed.emit(True)
        self.log_message.emit("Starting...\n")
        self._start_next_command()

    @Slot()
    def stop_pipeline(self) -> None:
        if self._process is None:
            return

        self._stopping = True
        self.log_message.emit("Stopping...\n")

        if os.name == "nt":
            self._process.kill()
        else:
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                try:
                    pid = self._process.processId()
                    if pid:
                        os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
                self._process.kill()

    def _start_next_command(self) -> None:
        if not self._queue:
            self._cleanup_process()
            self.busy_changed.emit(False)
            self.stage_changed.emit("Finished")
            self.pipeline_finished.emit(True, "Finished successfully.")
            return

        stage_name, command = self._queue.pop(0)
        self.stage_changed.emit(stage_name)
        self.log_message.emit(f"=== {stage_name} ===\n")
        self.log_message.emit("$ " + " ".join(command) + "\n\n")

        process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessEnvironment(self._build_process_environment())
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._handle_finished)
        process.errorOccurred.connect(self._handle_error)

        self._process = process
        process.start(command[0], command[1:])

    def _build_process_environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH", "")
        project_root = str(PROJECT_ROOT)
        if existing:
            env.insert("PYTHONPATH", project_root + os.pathsep + existing)
        else:
            env.insert("PYTHONPATH", project_root)
        return env

    @Slot()
    def _read_output(self) -> None:
        if self._process is None:
            return
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        if not data:
            return
        self._output_buffer += data
        newline_pos = self._output_buffer.rfind("\n")
        if newline_pos != -1:
            complete = self._output_buffer[: newline_pos + 1]
            self._output_buffer = self._output_buffer[newline_pos + 1 :]
            self.log_message.emit(complete)

    def _flush_output_buffer(self) -> None:
        if self._process is not None:
            remaining = self._process.readAllStandardOutput().data().decode(errors="replace")
            if remaining:
                self._output_buffer += remaining
        if self._output_buffer:
            self.log_message.emit(self._output_buffer)
            self._output_buffer = ""

    @Slot(int, QProcess.ExitStatus)
    def _handle_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self._process is None:
            return

        self._flush_output_buffer()

        if self._stopping:
            self._cleanup_process()
            self.busy_changed.emit(False)
            self.stage_changed.emit("Stopped")
            self.pipeline_finished.emit(False, "Pipeline stopped by user.")
            return

        if exit_code != 0:
            self._cleanup_process()
            self.busy_changed.emit(False)
            self.stage_changed.emit("Failed")
            self.pipeline_finished.emit(False, f"Stage failed with exit code {exit_code}.")
            return

        self.log_message.emit("\n")
        self._cleanup_process()
        self._start_next_command()

    @Slot(QProcess.ProcessError)
    def _handle_error(self, error: QProcess.ProcessError) -> None:
        self._cleanup_process()
        self.busy_changed.emit(False)
        self.stage_changed.emit("Error")
        self.pipeline_finished.emit(False, f"Failed to start or run process: {error}.")

    def _cleanup_process(self) -> None:
        if self._process is None:
            return
        self._process.deleteLater()
        self._process = None
        self._output_buffer = ""
