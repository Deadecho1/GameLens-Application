from __future__ import annotations

from PySide6.QtCore import Qt, Slot
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
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from app_core.formatting import format_seconds
from .analytics_service import AnalyticsService
from .config import (
    APP_NAME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
)
from .models import DashboardStats, GameInfo, RunSummary, VersionInfo
from .process_clips_dialog import ProcessClipsDialog
from .protocols import AnalyticsReader, GameRepo
from .repository import GameRepository
from .run_details_dialog import RunDetailsDialog
from .widgets import ResponsiveFontMixin, populate_combo_restoring_selection


class MainWindow(ResponsiveFontMixin, QMainWindow):
    def __init__(
        self,
        repo: GameRepo | None = None,
        analytics: AnalyticsReader | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        from .config import GAMES_ROOT
        self._repo: GameRepo = repo or GameRepository(root_dir=GAMES_ROOT)
        self._analytics: AnalyticsReader = analytics or AnalyticsService()
        self._games: list[GameInfo] = []
        self._versions: list[VersionInfo] = []
        self._current_runs: list[RunSummary] = []

        self._setup_font_timer()
        self._build_ui()
        self._connect_signals()
        self._load_games()
        self._apply_responsive_fonts()

    def _font_scale_params(self) -> tuple:
        return MIN_FONT_SIZE, MAX_FONT_SIZE, 900, 700, 78.0, 45.0

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Top row: Game selector + Version selector
        top_row = QHBoxLayout()

        game_group = QGroupBox("Game")
        game_layout = QHBoxLayout(game_group)
        self.game_combo = QComboBox()
        self.add_game_button = QPushButton("Add game")
        game_layout.addWidget(self.game_combo, 1)
        game_layout.addWidget(self.add_game_button)

        version_group = QGroupBox("Version")
        version_layout = QHBoxLayout(version_group)
        self.version_combo = QComboBox()
        self.add_version_button = QPushButton("Add version")
        self.process_button = QPushButton("Process clips")
        version_layout.addWidget(self.version_combo, 1)
        version_layout.addWidget(self.add_version_button)
        version_layout.addWidget(self.process_button)

        top_row.addWidget(game_group)
        top_row.addWidget(version_group)

        # Dashboard Overview
        dashboard_group = QGroupBox("Dashboard Overview")
        dashboard_layout = QGridLayout(dashboard_group)

        self.total_runs_value = QLabel("0")
        self.avg_duration_value = QLabel("0s")
        self.max_duration_value = QLabel("0s")
        self.popular_item_value = QLabel("—")

        dashboard_layout.addWidget(self._make_stat_card("Runs", self.total_runs_value), 0, 0)
        dashboard_layout.addWidget(self._make_stat_card("Average Duration", self.avg_duration_value), 0, 1)
        dashboard_layout.addWidget(self._make_stat_card("Longest Run", self.max_duration_value), 0, 2)
        dashboard_layout.addWidget(self._make_stat_card("Most Popular Item", self.popular_item_value), 0, 3)

        # Runs table
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
        self.game_combo.activated.connect(self._on_game_selected)
        self.version_combo.activated.connect(self._on_version_selected)
        self.add_game_button.clicked.connect(self._add_game)
        self.add_version_button.clicked.connect(self._add_version)
        self.process_button.clicked.connect(self._open_process_dialog)
        self.runs_table.cellDoubleClicked.connect(self._on_run_double_clicked)

    def _load_games(self) -> None:
        current_game = self.game_combo.currentText().strip()
        self._games = self._repo.list_games()
        populate_combo_restoring_selection(
            self.game_combo,
            [g.name for g in self._games],
            current_game,
        )
        self._load_versions()

    def _load_versions(self) -> None:
        game = self._current_game()
        current_version = self.version_combo.currentText().strip()
        self._versions = self._repo.list_versions(game) if game is not None else []
        populate_combo_restoring_selection(
            self.version_combo,
            [v.name for v in self._versions],
            current_version,
        )
        self._refresh_dashboard()

    def _current_game(self) -> GameInfo | None:
        index = self.game_combo.currentIndex()
        if index < 0 or index >= len(self._games):
            return None
        return self._games[index]

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
        self.popular_item_value.setText("—")

    def _set_stats(self, stats: DashboardStats) -> None:
        self.total_runs_value.setText(str(stats.total_runs))
        self.avg_duration_value.setText(format_seconds(stats.average_run_duration_seconds))
        self.max_duration_value.setText(format_seconds(stats.max_run_duration_seconds))
        self.popular_item_value.setText(stats.most_popular_item)

    def _populate_runs(self, runs: list[RunSummary]) -> None:
        self.runs_table.clearContents()
        self.runs_table.setRowCount(len(runs))

        for row, run in enumerate(runs):
            self.runs_table.setItem(row, 0, QTableWidgetItem(run.run_name))
            self.runs_table.setItem(row, 1, QTableWidgetItem(format_seconds(run.duration_seconds)))

        self.runs_table.resizeRowsToContents()
        self.runs_table.viewport().update()

    @Slot()
    def _add_game(self) -> None:
        game_name, ok = QInputDialog.getText(self, APP_NAME, "Enter new game name:")
        if not ok:
            return

        game_name = game_name.strip()
        if not game_name:
            QMessageBox.warning(self, APP_NAME, "Game name cannot be empty.")
            return

        if any(g.name == game_name for g in self._games):
            QMessageBox.warning(self, APP_NAME, f"Game '{game_name}' already exists.")
            return

        self._repo.ensure_game(game_name)
        self._load_games()

        index = self.game_combo.findText(game_name)
        if index >= 0:
            self.game_combo.setCurrentIndex(index)
            self._load_versions()

    @Slot()
    def _add_version(self) -> None:
        game = self._current_game()
        if game is None:
            QMessageBox.warning(self, APP_NAME, "Please add and select a game first.")
            return

        version_name, ok = QInputDialog.getText(self, APP_NAME, "Enter new version name:")
        if not ok:
            return

        version_name = version_name.strip()
        if not version_name:
            QMessageBox.warning(self, APP_NAME, "Version name cannot be empty.")
            return

        if any(v.name == version_name for v in self._versions):
            QMessageBox.warning(self, APP_NAME, f"Version '{version_name}' already exists.")
            return

        self._repo.ensure_version(game, version_name)
        self._load_versions()

        index = self.version_combo.findText(version_name)
        if index >= 0:
            self.version_combo.setCurrentIndex(index)
            self._refresh_dashboard()

    @Slot()
    def _open_process_dialog(self) -> None:
        version = self._current_version()
        if version is None:
            QMessageBox.warning(self, APP_NAME, "Please add and select a game and version first.")
            return

        dialog = ProcessClipsDialog(version, self)
        dialog.processing_completed.connect(self._refresh_dashboard)
        dialog.exec()

    @Slot(int)
    def _on_game_selected(self, _: int) -> None:
        self._load_versions()

    @Slot(int)
    def _on_version_selected(self, _: int) -> None:
        self._refresh_dashboard()

    @Slot(int, int)
    def _on_run_double_clicked(self, row: int, _: int) -> None:
        if row < 0 or row >= len(self._current_runs):
            return

        version = self._current_version()
        if version is None:
            return

        run_summary = self._current_runs[row]
        details = self._analytics.load_run_details(version, run_summary)
        dialog = RunDetailsDialog(details, self)
        dialog.exec()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_font_update()

    def _apply_responsive_fonts(self) -> None:
        min_fs, max_fs, min_w, min_h, w_ratio, h_ratio = self._font_scale_params()
        width = max(self.width(), min_w)
        height = max(self.height(), min_h)
        point_size = max(min_fs, min(max_fs, int(min(width / w_ratio, height / h_ratio))))

        self._apply_font_recursive(self.centralWidget(), point_size)

        stat_font = QFont(self.font())
        stat_font.setPointSize(point_size + 6)
        stat_font.setBold(True)
        for label in [
            self.total_runs_value,
            self.avg_duration_value,
            self.max_duration_value,
            self.popular_item_value,
        ]:
            label.setFont(stat_font)

        title_font = QFont(self.font())
        title_font.setPointSize(point_size + 1)
        for group_box in self.findChildren(QGroupBox):
            group_box.setFont(title_font)
