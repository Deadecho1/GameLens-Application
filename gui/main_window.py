from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from .analytics_service import AnalyticsService
from .config import (
    APP_NAME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
)
from .models import DashboardStats, RunSummary, VersionInfo
from .process_clips_dialog import ProcessClipsDialog
from .repository import VersionRepository


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self._repo = VersionRepository()
        self._analytics = AnalyticsService()
        self._versions: list[VersionInfo] = []
        self._current_runs: list[RunSummary] = []
        self._run_tabs: dict[str, QWidget] = {}

        self._font_update_timer = QTimer(self)
        self._font_update_timer.setSingleShot(True)
        self._font_update_timer.timeout.connect(self._apply_responsive_fonts)

        self._build_ui()
        self._connect_signals()
        self._load_versions()
        self._apply_responsive_fonts()
        self._refresh_dashboard()

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.dashboard_tab = QWidget()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")

        root = QVBoxLayout(self.dashboard_tab)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        top_row = QHBoxLayout()

        version_group = QGroupBox("Version")
        version_layout = QHBoxLayout(version_group)

        self.version_combo = QComboBox()
        self.add_version_button = QPushButton("Add version")
        self.process_button = QPushButton("Process clips")

        version_layout.addWidget(QLabel("Selected version"))
        version_layout.addWidget(self.version_combo, 1)
        version_layout.addWidget(self.add_version_button)
        version_layout.addWidget(self.process_button)

        top_row.addWidget(version_group)

        dashboard_group = QGroupBox("Dashboard Overview")
        dashboard_layout = QGridLayout(dashboard_group)

        self.total_runs_value = QLabel("0")
        self.avg_duration_value = QLabel("0s")
        self.max_duration_value = QLabel("0s")
        self.total_choices_value = QLabel("0")

        dashboard_layout.addWidget(self._make_stat_card("Runs", self.total_runs_value), 0, 0)
        dashboard_layout.addWidget(self._make_stat_card("Average Duration", self.avg_duration_value), 0, 1)
        dashboard_layout.addWidget(self._make_stat_card("Longest Run", self.max_duration_value), 0, 2)
        dashboard_layout.addWidget(self._make_stat_card("Total Selections", self.total_choices_value), 0, 3)

        runs_group = QGroupBox("Runs")
        runs_layout = QVBoxLayout(runs_group)

        self.runs_table = QTableWidget(0, 2)
        self.runs_table.setHorizontalHeaderLabels(["Run ID", "Duration"])
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.runs_table.setSelectionMode(QTableWidget.SingleSelection)
        self.runs_table.setAlternatingRowColors(True)

        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        runs_layout.addWidget(self.runs_table)

        root.addLayout(top_row)
        root.addWidget(dashboard_group)
        root.addWidget(runs_group, 1)

    def _make_stat_card(self, title: str, value_label: QLabel) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)
        return box

    def _connect_signals(self) -> None:
        self.version_combo.activated.connect(self._on_version_selected)
        self.add_version_button.clicked.connect(self._add_version)
        self.process_button.clicked.connect(self._open_process_dialog)
        self.runs_table.cellDoubleClicked.connect(self._on_run_double_clicked)

    def _load_versions(self) -> None:
        current_name = self.version_combo.currentText().strip()
        self._versions = self._repo.list_versions()

        self.version_combo.blockSignals(True)
        self.version_combo.clear()
        for version in self._versions:
            self.version_combo.addItem(version.name)
        self.version_combo.blockSignals(False)

        if self._versions:
            index = self.version_combo.findText(current_name)
            self.version_combo.setCurrentIndex(index if index >= 0 else 0)

        self._refresh_dashboard()

    def _current_version(self) -> VersionInfo | None:
        index = self.version_combo.currentIndex()
        if index < 0 or index >= len(self._versions):
            return None
        return self._versions[index]

    @Slot()
    def _refresh_dashboard(self) -> None:
        version = self._current_version()
        if version is None:
            self._current_runs = []
            self._set_stats_empty()
            self._populate_runs([])
            return

        stats = self._analytics.load_dashboard_stats(version)
        runs = self._analytics.load_run_summaries(version)
        self._current_runs = runs
        self._set_stats(stats)
        self._populate_runs(runs)

    def _set_stats_empty(self) -> None:
        self.total_runs_value.setText("0")
        self.avg_duration_value.setText("0s")
        self.max_duration_value.setText("0s")
        self.total_choices_value.setText("0")

    def _set_stats(self, stats: DashboardStats) -> None:
        self.total_runs_value.setText(str(stats.total_runs))
        self.avg_duration_value.setText(self._format_seconds(stats.average_run_duration_seconds))
        self.max_duration_value.setText(self._format_seconds(stats.max_run_duration_seconds))
        self.total_choices_value.setText(str(stats.total_choices))

    def _populate_runs(self, runs: list[RunSummary]) -> None:
        self.runs_table.clearContents()
        self.runs_table.setRowCount(len(runs))

        for row, run in enumerate(runs):
            self.runs_table.setItem(row, 0, QTableWidgetItem(run.run_name))
            self.runs_table.setItem(row, 1, QTableWidgetItem(self._format_seconds(run.duration_seconds)))

        self.runs_table.resizeRowsToContents()
        self.runs_table.viewport().update()

    def _format_seconds(self, seconds: float) -> str:
        total = int(seconds)
        minutes, sec = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {sec}s"
        if minutes > 0:
            return f"{minutes}m {sec}s"
        return f"{sec}s"

    @Slot()
    def _add_version(self) -> None:
        version_name, ok = QInputDialog.getText(self, APP_NAME, "Enter new version name:")
        if not ok:
            return

        version_name = version_name.strip()
        if not version_name:
            QMessageBox.warning(self, APP_NAME, "Version name cannot be empty.")
            return

        if any(version.name == version_name for version in self._versions):
            QMessageBox.warning(self, APP_NAME, f"Version '{version_name}' already exists.")
            return

        self._repo.ensure_version(version_name)
        self._load_versions()

        index = self.version_combo.findText(version_name)
        if index >= 0:
            self.version_combo.setCurrentIndex(index)
            self._refresh_dashboard()

    @Slot()
    def _open_process_dialog(self) -> None:
        version = self._current_version()
        if version is None:
            QMessageBox.warning(self, APP_NAME, "Please add and select a version first.")
            return

        dialog = ProcessClipsDialog(version, self)
        dialog.processing_completed.connect(self._refresh_dashboard)
        dialog.exec()

    @Slot(int)
    def _on_version_selected(self, index: int) -> None:
        self.version_combo.setCurrentIndex(index)
        self.version_combo.hidePopup()
        self._refresh_dashboard()

    @Slot(int, int)
    def _on_run_double_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._current_runs):
            return

        run = self._current_runs[row]
        self._open_run_tab(run)

    def _open_run_tab(self, run: RunSummary) -> None:
        run_key = run.run_name

        existing_tab = self._run_tabs.get(run_key)
        if existing_tab is not None:
            self.tabs.setCurrentWidget(existing_tab)
            return

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(run.run_name)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        duration_label = QLabel(f"Duration: {self._format_seconds(run.duration_seconds)}")

        layout.addWidget(title)
        layout.addWidget(duration_label)
        layout.addStretch(1)

        self.tabs.addTab(tab, run.run_name)
        self.tabs.setCurrentWidget(tab)
        self._run_tabs[run_key] = tab

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._font_update_timer.start(50)

    def _apply_font_recursive(self, widget: QWidget, point_size: int) -> None:
        font = QFont(widget.font())
        font.setPointSize(point_size)
        widget.setFont(font)
        for child in widget.findChildren(QWidget):
            child_font = QFont(child.font())
            child_font.setPointSize(point_size)
            child.setFont(child_font)

    def _apply_responsive_fonts(self) -> None:
        width = max(self.width(), 900)
        height = max(self.height(), 700)
        point_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, int(min(width / 78, height / 45))))

        self._apply_font_recursive(self.centralWidget(), point_size)

        stat_font = QFont(self.font())
        stat_font.setPointSize(point_size + 6)
        stat_font.setBold(True)
        for label in [
            self.total_runs_value,
            self.avg_duration_value,
            self.max_duration_value,
            self.total_choices_value,
        ]:
            label.setFont(stat_font)

        title_font = QFont(self.font())
        title_font.setPointSize(point_size + 1)
        for group_box in self.findChildren(QGroupBox):
            group_box.setFont(title_font)