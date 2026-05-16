from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_core.analytics import AnalyticsService
from app_core.formatting import format_seconds
from app_core.models import DashboardStats, GameInfo, RunSummary, VersionInfo

from .config import (
    APP_NAME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
)
from .models import PipelineConfig
from .pipeline_runner import PipelineRunner
from .process_clips_dialog import ProcessClipsDialog
from .storage.base import StorageBackend
from .storage.local import LocalSQLiteBackend
from .sync_worker import SyncWorker
from .tuning_config import TUNING_MODEL_CONFIG_BY_ID
from .tuning_runner import TuningRunner
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
        self._ipc_ui_state: dict = {
            "activeMainTab": "workflow",
            "workflowStep": 1,
            "completionCelebrationActive": False,
            "changePicker": None,
            "addGameModalOpen": False,
            "addVersionModalOpen": False,
            "newGameNameDraft": "",
            "newVersionNameDraft": "",
            "analyticsSubTab": "general",
        }
        self._processing_state: dict = {
            "pipelinePath": "",
            "videoFiles": [],
            "videoFilePaths": [],
            "status": "idle",
            "selectedModel": "base",
            "logs": ["[INFO] System ready..."],
        }
        self._pipeline_runner = PipelineRunner()
        self._auth_state: dict = {
            "loggedIn": False,
            "email": None,
            "userId": None,
            "syncStatus": "idle",
            "syncMessage": "",
        }
        self._active_backend: StorageBackend = LocalSQLiteBackend()
        self._sync_worker: SyncWorker | None = None
        self._tuning_state: dict = {
            "status": "idle",
            "logs": [],
        }
        self._tuning_runner = TuningRunner()

        self._setup_font_timer()
        self._build_ui()
        self._connect_signals()
        self._connect_pipeline_signals()
        self._connect_tuning_signals()
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

        dashboard_layout.addWidget(
            self._make_stat_card("Runs", self.total_runs_value), 0, 0
        )
        dashboard_layout.addWidget(
            self._make_stat_card("Average Duration", self.avg_duration_value), 0, 1
        )
        dashboard_layout.addWidget(
            self._make_stat_card("Longest Run", self.max_duration_value), 0, 2
        )
        dashboard_layout.addWidget(
            self._make_stat_card("Most Popular Item", self.popular_item_value), 0, 3
        )

        # Runs table
        runs_group = QGroupBox("Runs")
        runs_layout = QVBoxLayout(runs_group)

        self.runs_table = QTableWidget(0, 2)
        self.runs_table.setHorizontalHeaderLabels(["Run ID", "Duration"])
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.runs_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.runs_table.setAlternatingRowColors(True)

        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)

        runs_layout.addWidget(self.runs_table)

        root.addLayout(top_row)
        root.addWidget(dashboard_group)
        root.addWidget(runs_group, 1)

    def _make_stat_card(self, title: str, value_label: QLabel) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)
        return box

    def _connect_signals(self) -> None:
        self.game_combo.activated.connect(self._on_game_selected)
        self.version_combo.activated.connect(self._on_version_selected)
        self.add_game_button.clicked.connect(self._add_game)
        self.add_version_button.clicked.connect(self._add_version)
        self.process_button.clicked.connect(self._open_process_dialog)
        self.runs_table.cellDoubleClicked.connect(self._on_run_double_clicked)

    def _connect_pipeline_signals(self) -> None:
        self._pipeline_runner.log_message.connect(self._on_pipeline_log)
        self._pipeline_runner.stage_changed.connect(self._on_pipeline_stage_changed)
        self._pipeline_runner.pipeline_finished.connect(self._on_pipeline_finished)

    def _connect_tuning_signals(self) -> None:
        self._tuning_runner.log_message.connect(self._on_tuning_log)
        self._tuning_runner.tuning_finished.connect(self._on_tuning_finished)
        self._tuning_runner.busy_changed.connect(self._on_tuning_busy_changed)

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

    def _format_mm_ss(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        minutes, sec = divmod(total, 60)
        return f"{minutes:02d}:{sec:02d}"

    def _build_dashboard_payload(self, version: VersionInfo | None) -> dict:
        if version is None:
            return {"items": [], "bosses": [], "runsHistory": []}

        runs = self._analytics.load_run_summaries(version)

        item_counter: dict[str, int] = {}
        for run in runs:
            for item in run.selected_items:
                cleaned = item.strip() or "Unknown"
                item_counter[cleaned] = item_counter.get(cleaned, 0) + 1

        items = []
        sorted_items = sorted(item_counter.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_items[0][1] if sorted_items else 1
        for index, (name, count) in enumerate(sorted_items, start=1):
            popularity = int(round((count / max_count) * 100)) if max_count > 0 else 0
            items.append(
                {
                    "id": index,
                    "name": name,
                    "popularity": popularity,
                    "impact": "Medium",
                    "category": "utility",
                    "logicTag": "Detected from run choices",
                    "rarity": "Common",
                    "avgPossessionMinutes": 0,
                }
            )

        item_id_by_name = {item["name"]: item["id"] for item in items}
        runs_history = []
        for idx, run in enumerate(runs, start=1):
            loadout = [
                item_id_by_name[name]
                for name in run.selected_items
                if name in item_id_by_name
            ][:3]
            runs_history.append(
                {
                    "id": f"RUN-{idx:03d}",
                    "date": str(date.today()),
                    "duration": self._format_mm_ss(run.duration_seconds),
                    "bossEncounters": [
                        {
                            "bossId": 1,
                            "lifespan": self._format_mm_ss(run.duration_seconds),
                            "loadout": loadout,
                        }
                    ]
                    if items
                    else [],
                }
            )

        bosses = []
        if runs_history:
            bosses.append(
                {
                    "id": 1,
                    "name": "Session Boss",
                    "lifespan": runs_history[0]["duration"],
                    "status": "Defeated",
                    "globalLifespanSamples": [r["duration"] for r in runs_history],
                    "gearSynergies": [],
                    "itemEffectiveness": [],
                }
            )

        return {
            "items": items,
            "bosses": bosses,
            "runsHistory": runs_history,
        }

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
        self.avg_duration_value.setText(
            format_seconds(stats.average_run_duration_seconds)
        )
        self.max_duration_value.setText(format_seconds(stats.max_run_duration_seconds))
        self.popular_item_value.setText(stats.most_popular_item)

    @Slot(str)
    def _on_pipeline_log(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self._processing_state["logs"].append(line)

    @Slot(str)
    def _on_pipeline_stage_changed(self, stage: str) -> None:
        if stage in {"Finished"}:
            self._processing_state["status"] = "completed"
        elif stage in {"Stopped"}:
            self._processing_state["status"] = "stopped"
        elif stage in {"Failed", "Error"}:
            self._processing_state["status"] = "stopped"
        else:
            self._processing_state["status"] = "running"

    @Slot(bool, str)
    def _on_pipeline_finished(self, success: bool, message: str) -> None:
        self._processing_state["logs"].append(message)
        self._processing_state["status"] = "completed" if success else "stopped"
        if success:
            self._refresh_dashboard()
            if self._auth_state.get("loggedIn"):
                self._start_sync()

    def _populate_runs(self, runs: list[RunSummary]) -> None:
        self.runs_table.clearContents()
        self.runs_table.setRowCount(len(runs))

        for row, run in enumerate(runs):
            self.runs_table.setItem(row, 0, QTableWidgetItem(run.run_name))
            self.runs_table.setItem(
                row, 1, QTableWidgetItem(format_seconds(run.duration_seconds))
            )

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

        version_name, ok = QInputDialog.getText(
            self, APP_NAME, "Enter new version name:"
        )
        if not ok:
            return

        version_name = version_name.strip()
        if not version_name:
            QMessageBox.warning(self, APP_NAME, "Version name cannot be empty.")
            return

        if any(v.name == version_name for v in self._versions):
            QMessageBox.warning(
                self, APP_NAME, f"Version '{version_name}' already exists."
            )
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
            QMessageBox.warning(
                self, APP_NAME, "Please add and select a game and version first."
            )
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

    # ---- IPC public API ----
    def patch_ui_from_ipc(self, ui_patch: dict) -> None:
        self._ipc_ui_state = {**self._ipc_ui_state, **ui_patch}

    def save_settings_from_ipc(self, openai_key: str) -> None:
        setup = self._ipc_ui_state.get("setup", {})
        user = {**setup.get("user", {}), "openAiKey": openai_key}
        self._ipc_ui_state = {**self._ipc_ui_state, "setup": {**setup, "user": user}}

    def select_game_from_ipc(self, game_name: str) -> None:
        index = self.game_combo.findText(game_name)
        if index >= 0:
            self.game_combo.setCurrentIndex(index)
            self._load_versions()

    def select_version_from_ipc(self, version_name: str) -> None:
        index = self.version_combo.findText(version_name)
        if index >= 0:
            self.version_combo.setCurrentIndex(index)
            self._refresh_dashboard()

    def add_game_from_ipc(self, game_name: str) -> None:
        cleaned = game_name.strip()
        if not cleaned:
            raise ValueError("Game name cannot be empty.")
        self._repo.ensure_game(cleaned)
        self._load_games()
        self.select_game_from_ipc(cleaned)

    def add_version_from_ipc(self, game_name: str, version_name: str) -> None:
        cleaned_game = game_name.strip()
        cleaned_version = version_name.strip()
        if not cleaned_game or not cleaned_version:
            raise ValueError("game_name and version_name are required")
        game = next(
            (g for g in self._repo.list_games() if g.name == cleaned_game), None
        )
        if game is None:
            game = self._repo.ensure_game(cleaned_game)
        self._repo.ensure_version(game, cleaned_version)
        self._load_games()
        self.select_game_from_ipc(cleaned_game)
        self.select_version_from_ipc(cleaned_version)

    def stage_folder_from_ipc(self, pipeline_path: str) -> None:
        path = Path(pipeline_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid folder: {pipeline_path}")
        files = sorted([p.name for p in path.glob("*.mp4")])
        self._processing_state["pipelinePath"] = str(path)
        self._processing_state["videoFiles"] = files
        self._processing_state["videoFilePaths"] = [str(path / name) for name in files]

    def stage_files_from_ipc(
        self,
        file_names: list[str],
        pipeline_path: str | None = None,
        file_paths: list[str] | None = None,
    ) -> None:
        cleaned = [str(name).strip() for name in file_names if str(name).strip()]
        self._processing_state["videoFiles"] = sorted(list(dict.fromkeys(cleaned)))

        cleaned_paths = [str(p).strip() for p in (file_paths or []) if str(p).strip()]
        self._processing_state["videoFilePaths"] = sorted(
            list(dict.fromkeys(cleaned_paths))
        )

        if pipeline_path is not None:
            self._processing_state["pipelinePath"] = str(pipeline_path)

    def set_pipeline_path_from_ipc(self, pipeline_path: str) -> None:
        path = Path(pipeline_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid folder: {pipeline_path}")
        self._processing_state["pipelinePath"] = str(path)

    def run_pipeline_from_ipc(self) -> None:
        game = self._current_game()
        version = self._current_version()
        if version is None:
            raise ValueError("Select a game version before running the pipeline")

        pipeline_path = str(self._processing_state.get("pipelinePath") or "").strip()
        if not pipeline_path:
            raise ValueError("Pipeline path is not set")

        video_dir = Path(pipeline_path)
        if not video_dir.exists() or not video_dir.is_dir():
            raise ValueError(f"Pipeline path is not a valid folder: {pipeline_path}")

        staged_file_paths = self._processing_state.get("videoFilePaths") or []
        if isinstance(staged_file_paths, list) and staged_file_paths:
            parent_dirs = sorted(
                {
                    str(Path(p).resolve().parent)
                    for p in staged_file_paths
                    if Path(p).exists()
                }
            )
            if len(parent_dirs) == 1:
                video_dir = Path(parent_dirs[0])
                self._processing_state["pipelinePath"] = str(video_dir)
            elif len(parent_dirs) > 1:
                raise ValueError(
                    "Staged files come from multiple folders. Please stage files from one folder only, "
                    "or choose a single pipeline path folder."
                )

        openai_key = (
            self._ipc_ui_state.get("setup", {})
            .get("user", {})
            .get("openAiKey", "")
            or ""
        ).strip()
        if not openai_key:
            raise ValueError("OpenAI API key is not set. Add it in Settings before running the pipeline.")

        selected_model = self._processing_state.get("selectedModel", "base")
        model_dir = None
        if selected_model != "base":
            from app_core.config import AppConfig as _AC
            _cfg = _AC.load()
            candidate = _cfg.models_dir / "finetuned" / selected_model
            if candidate.exists():
                model_dir = candidate
        from app_core.config import AppConfig as _AC
        _cfg = _AC.load()
        current_user_id = str(self._auth_state.get("userId") or 0)
        config = PipelineConfig(
            video_dir=video_dir,
            event_json_dir=version.event_json_dir,
            run_json_dir=version.run_json_dir,
            openai_api_key=openai_key,
            game_name=game.name if game else "",
            version_name=version.name,
            user_id=current_user_id,
            collector_url=_cfg.collector_base_url,
            model_dir=model_dir,
        )
        self._processing_state["status"] = "running"
        self._processing_state["logs"].append("[RUN] Started from Electron")
        self._pipeline_runner.start_pipeline(config)

    def stop_pipeline_from_ipc(self) -> None:
        self._pipeline_runner.stop_pipeline()

    def clear_processing_logs_from_ipc(self) -> None:
        self._processing_state["logs"] = []

    def set_model_from_ipc(self, model: str) -> None:
        self._processing_state["selectedModel"] = model

    # ---- Tuning signal handlers ----
    @Slot(str)
    def _on_tuning_log(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self._tuning_state["logs"].append(line)

    @Slot(bool)
    def _on_tuning_busy_changed(self, busy: bool) -> None:
        if not busy and self._tuning_state["status"] == "running":
            pass  # status set by _on_tuning_finished

    @Slot(bool, str)
    def _on_tuning_finished(self, success: bool, message: str) -> None:
        self._tuning_state["logs"].append(message)
        self._tuning_state["status"] = "completed" if success else "stopped"

    # ---- Tuning IPC public API ----
    def start_tuning_from_ipc(self, videos: list, enabled_model_ids: list, model_name: str) -> None:
        from app_core.config import AppConfig
        app_config = AppConfig.load()
        finetuned_dir = app_config.models_dir / "finetuned"
        finetuned_dir.mkdir(parents=True, exist_ok=True)
        self._tuning_state["status"] = "running"
        self._tuning_state["logs"] = []
        self._tuning_runner.start_tuning(
            videos=videos,
            enabled_model_ids=enabled_model_ids,
            model_name=model_name,
            finetuned_models_dir=finetuned_dir,
            base_model_dir=app_config.event_detector_model_dir,
            boss_model_path=app_config.boss_model_path,
        )

    def stop_tuning_from_ipc(self) -> None:
        self._tuning_runner.stop_tuning()

    def clear_tuning_logs_from_ipc(self) -> None:
        self._tuning_state["logs"] = []

    def get_finetuned_models_from_ipc(self) -> list[dict]:
        from app_core.config import AppConfig
        app_config = AppConfig.load()
        finetuned_dir = app_config.models_dir / "finetuned"
        if not finetuned_dir.exists():
            return []
        models = []
        for d in sorted(finetuned_dir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            # name format: {modelName}_{modelId}
            for model_id in TUNING_MODEL_CONFIG_BY_ID:
                suffix = f"_{model_id}"
                if name.endswith(suffix):
                    model_name = name[: -len(suffix)]
                    models.append({"name": model_name, "modelId": model_id, "dirName": name})
                    break
        return models

    # ---- Auth IPC public API ----

    def login_from_ipc(self, email: str) -> dict:
        import requests
        from app_core.config import AppConfig
        cfg = AppConfig.load()
        resp = requests.post(
            f"{cfg.collector_base_url}/api/v1/auth/login",
            json={"email": email},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        user_id = int(data["user_id"])

        from .storage.remote import RemoteCollectorBackend
        from app_core.local_storage import LOCAL_USER_ID, open_local_db
        self._active_backend = RemoteCollectorBackend(cfg.collector_base_url, user_id)
        self._auth_state = {
            "loggedIn": True,
            "email": email,
            "userId": user_id,
            "syncStatus": "syncing",
            "syncMessage": "Starting sync...",
        }

        # Re-attribute any offline (user_id=0) runs to the now-logged-in user
        # so pipeline runs recorded while offline remain visible after login.
        if user_id != LOCAL_USER_ID:
            try:
                conn = open_local_db()
                conn.execute(
                    "UPDATE dash_games SET user_id = ? WHERE user_id = ?",
                    (user_id, LOCAL_USER_ID),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        self._start_sync()
        return dict(self._auth_state)

    def logout_from_ipc(self) -> None:
        if self._sync_worker and self._sync_worker.isRunning():
            self._sync_worker.quit()
            self._sync_worker.wait(2000)

        prev_user_id = self._auth_state.get("userId")
        self._active_backend = LocalSQLiteBackend()
        self._auth_state = {
            "loggedIn": False,
            "email": None,
            "userId": None,
            "syncStatus": "idle",
            "syncMessage": "",
        }

        # Return local game records to user_id=0 so offline mode can see them.
        if prev_user_id and prev_user_id != 0:
            try:
                from app_core.local_storage import LOCAL_USER_ID, open_local_db
                conn = open_local_db()
                conn.execute(
                    "UPDATE dash_games SET user_id = ? WHERE user_id = ?",
                    (LOCAL_USER_ID, prev_user_id),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

    def _start_sync(self) -> None:
        if self._sync_worker and self._sync_worker.isRunning():
            return  # already syncing
        from .storage.remote import RemoteCollectorBackend
        if not isinstance(self._active_backend, RemoteCollectorBackend):
            return
        self._auth_state["syncStatus"] = "syncing"
        self._auth_state["syncMessage"] = "Starting sync..."
        self._sync_worker = SyncWorker(self._active_backend)
        self._sync_worker.sync_progress.connect(self._on_sync_progress)
        self._sync_worker.sync_finished.connect(self._on_sync_finished)
        self._sync_worker.start()

    @Slot(str)
    def _on_sync_progress(self, message: str) -> None:
        self._auth_state["syncMessage"] = message

    @Slot(bool, str)
    def _on_sync_finished(self, success: bool, message: str) -> None:
        self._auth_state["syncStatus"] = "done" if success else "error"
        self._auth_state["syncMessage"] = message
        if success:
            self._active_backend = LocalSQLiteBackend()

    # ---- Dashboard IPC public API ----

    def get_dashboard_items_from_ipc(self, game_name: str, version_name: str | None) -> list[dict]:
        user_id = self._auth_state.get("userId") or 0
        return self._active_backend.get_items(user_id, game_name, version_name or None)

    def get_dashboard_bosses_from_ipc(self, game_name: str, version_name: str | None) -> list[dict]:
        user_id = self._auth_state.get("userId") or 0
        return self._active_backend.get_bosses(user_id, game_name, version_name or None)

    def get_dashboard_runs_from_ipc(self, game_name: str, version_name: str | None) -> list[dict]:
        user_id = self._auth_state.get("userId") or 0
        return self._active_backend.get_runs(user_id, game_name, None)

    def get_dashboard_stats_from_ipc(self, game_name: str, version_name: str | None) -> dict:
        user_id = self._auth_state.get("userId") or 0
        return self._active_backend.get_stats(user_id, game_name, version_name or None)

    def get_frontend_state_from_ipc(self) -> dict:
        current_game = self._current_game()
        current_version = self._current_version()
        game_name = current_game.name if current_game else ""
        version_name = current_version.name if current_version else None
        user_id = self._auth_state.get("userId") or 0
        dashboard = {
            "items": self._active_backend.get_items(user_id, game_name, version_name),
            "bosses": self._active_backend.get_bosses(user_id, game_name, version_name),
            "runsHistory": self._active_backend.get_runs(user_id, game_name, None),
        }

        finetuned_models = self.get_finetuned_models_from_ipc()

        setup = {
            "games": [g.name for g in self._games],
            "versions": [v.name for v in self._versions],
            "selectedGame": current_game.name if current_game else "",
            "selectedVersion": current_version.name if current_version else "",
            "fineTunedModels": finetuned_models,
            "selectedModel": self._processing_state.get("selectedModel", "base"),
        }

        return {
            "ui": dict(self._ipc_ui_state),
            "setup": setup,
            "processing": dict(self._processing_state),
            "tuning": dict(self._tuning_state),
            "dashboard": dashboard,
            "auth": dict(self._auth_state),
        }

    def _apply_responsive_fonts(self) -> None:
        min_fs, max_fs, min_w, min_h, w_ratio, h_ratio = self._font_scale_params()
        width = max(self.width(), min_w)
        height = max(self.height(), min_h)
        point_size = max(
            min_fs, min(max_fs, int(min(width / w_ratio, height / h_ratio)))
        )

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
