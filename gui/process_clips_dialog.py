from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import APP_NAME, MAX_FONT_SIZE, MIN_FONT_SIZE
from .models import PipelineConfig, VersionInfo
from .pipeline_runner import PipelineRunner
from .widgets import ResponsiveFontMixin


class ProcessClipsDialog(ResponsiveFontMixin, QDialog):
    processing_completed = Signal()

    def __init__(self, version: VersionInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._version = version
        self.runner = PipelineRunner()

        self.setWindowTitle(f"{APP_NAME} - Process Clips - {version.name}")
        self.resize(1100, 800)

        self._setup_font_timer()
        self._build_ui()
        self._connect_signals()
        self._apply_responsive_fonts()

    def _font_scale_params(self) -> tuple:
        return MIN_FONT_SIZE, MAX_FONT_SIZE, 800, 600, 70.0, 42.0

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        version_group = QGroupBox("Target Version")
        version_layout = QHBoxLayout(version_group)
        version_layout.addWidget(QLabel(f"Processing into version: {self._version.name}"))
        version_layout.addStretch(1)

        config_group = QGroupBox("Pipeline")
        config_layout = QGridLayout(config_group)
        self.video_dir_edit = QLineEdit()
        self.video_browse_button = QPushButton("Browse...")
        config_layout.addWidget(QLabel("Video folder"), 0, 0)
        config_layout.addWidget(self.video_dir_edit, 0, 1)
        config_layout.addWidget(self.video_browse_button, 0, 2)
        config_layout.setColumnStretch(1, 1)

        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(120)
        self._video_count_label = QLabel("No videos found")
        config_layout.addWidget(self._video_count_label, 1, 0)
        config_layout.addWidget(self.video_list, 1, 1, 1, 2)

        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)
        self.only_events_checkbox = QCheckBox("Only events")
        self.only_export_checkbox = QCheckBox("Only export")
        self.verbose_checkbox = QCheckBox("Verbose")
        self.verbose_checkbox.setChecked(True)
        options_layout.addWidget(self.only_events_checkbox)
        options_layout.addWidget(self.only_export_checkbox)
        options_layout.addWidget(self.verbose_checkbox)
        options_layout.addStretch(1)

        actions_layout = QHBoxLayout()
        self.run_button = QPushButton("Run GameLens")
        self.stop_button = QPushButton("Stop")
        self.clear_log_button = QPushButton("Clear log")
        self.stop_button.setEnabled(False)
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(self.stop_button)
        actions_layout.addStretch(1)
        actions_layout.addWidget(self.clear_log_button)

        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)
        self.stage_value = QLabel("Idle")
        self.stage_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_layout.addWidget(QLabel("Current stage"), 0, 0)
        status_layout.addWidget(self.stage_value, 0, 1)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.log_output)

        root.addWidget(version_group)
        root.addWidget(config_group)
        root.addWidget(options_group)
        root.addLayout(actions_layout)
        root.addWidget(status_group)
        root.addWidget(log_group, 1)

    def _connect_signals(self) -> None:
        self.video_browse_button.clicked.connect(self._choose_video_dir)
        self.video_dir_edit.textChanged.connect(self._refresh_video_list)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.stop_button.clicked.connect(self.runner.stop_pipeline)
        self.clear_log_button.clicked.connect(self.log_output.clear)

        self.only_events_checkbox.toggled.connect(self._sync_mode_checkboxes)
        self.only_export_checkbox.toggled.connect(self._sync_mode_checkboxes)

        self.runner.log_message.connect(self._append_log)
        self.runner.stage_changed.connect(self.stage_value.setText)
        self.runner.pipeline_finished.connect(self._on_pipeline_finished)
        self.runner.busy_changed.connect(self._set_busy_state)

    @Slot()
    def _choose_video_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose video folder")
        if folder:
            self.video_dir_edit.setText(folder)

    @Slot(str)
    def _refresh_video_list(self, text: str) -> None:
        self.video_list.clear()
        folder = Path(text.strip())
        if not folder.is_dir():
            self._video_count_label.setText("No videos found")
            return
        videos = sorted(folder.glob("*.mp4"))
        for v in videos:
            self.video_list.addItem(v.name)
        count = len(videos)
        self._video_count_label.setText(f"{count} video{'s' if count != 1 else ''} found")

    @Slot(bool)
    def _sync_mode_checkboxes(self, checked: bool) -> None:
        sender = self.sender()
        if not checked:
            return
        if sender is self.only_events_checkbox:
            self.only_export_checkbox.setChecked(False)
        elif sender is self.only_export_checkbox:
            self.only_events_checkbox.setChecked(False)

    @Slot()
    def _on_run_clicked(self) -> None:
        config = self._build_config_from_ui()
        if config is None:
            return
        self.stage_value.setText("Preparing")
        self.runner.start_pipeline(config)

    def _build_config_from_ui(self) -> Optional[PipelineConfig]:
        video_dir_text = self.video_dir_edit.text().strip()
        if not video_dir_text:
            QMessageBox.warning(self, APP_NAME, "Please choose a video folder.")
            return None

        video_dir = Path(video_dir_text).resolve()
        if not video_dir.exists() or not video_dir.is_dir():
            QMessageBox.warning(self, APP_NAME, f"Invalid video folder:\n{video_dir}")
            return None

        self._version.event_json_dir.mkdir(parents=True, exist_ok=True)
        self._version.run_json_dir.mkdir(parents=True, exist_ok=True)

        return PipelineConfig(
            video_dir=video_dir,
            event_json_dir=self._version.event_json_dir,
            run_json_dir=self._version.run_json_dir,
            only_events=self.only_events_checkbox.isChecked(),
            only_export=self.only_export_checkbox.isChecked(),
            verbose=self.verbose_checkbox.isChecked(),
        )

    @Slot(str)
    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text.rstrip("\n"))
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(bool)
    def _set_busy_state(self, busy: bool) -> None:
        self.video_dir_edit.setEnabled(not busy)
        self.video_browse_button.setEnabled(not busy)
        self.video_list.setEnabled(not busy)
        self.only_events_checkbox.setEnabled(not busy)
        self.only_export_checkbox.setEnabled(not busy)
        self.verbose_checkbox.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_font_update()

    def _apply_responsive_fonts(self) -> None:
        min_fs, max_fs, min_w, min_h, w_ratio, h_ratio = self._font_scale_params()
        width = max(self.width(), min_w)
        height = max(self.height(), min_h)
        point_size = max(min_fs, min(max_fs, int(min(width / w_ratio, height / h_ratio))))

        self._apply_font_recursive(self, point_size)

        log_font = QFont(self.log_output.font())
        log_font.setPointSize(max(MIN_FONT_SIZE - 1, point_size - 1))
        self.log_output.setFont(log_font)

        title_font = QFont(self.font())
        title_font.setPointSize(point_size + 1)
        for group_box in self.findChildren(QGroupBox):
            group_box.setFont(title_font)

    @Slot(bool, str)
    def _on_pipeline_finished(self, success: bool, message: str) -> None:
        self._append_log(message)
        if success:
            self.processing_completed.emit()
            QMessageBox.information(self, APP_NAME, message)
        else:
            QMessageBox.warning(self, APP_NAME, message)
