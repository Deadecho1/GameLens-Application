from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app_core.analytics import AnalyticsService
from app_core.config import AppConfig

from .ipc_bridge import IpcBridge
from .main_window import MainWindow
from .repository import GameRepository


def main() -> None:
    config = AppConfig.load()
    repo = GameRepository(root_dir=config.games_root)
    analytics = AnalyticsService()

    app = QApplication(sys.argv)
    base_font = QFont()
    base_font.setPointSize(14)
    app.setFont(base_font)

    window = MainWindow(repo=repo, analytics=analytics)
    ipc = IpcBridge(window=window, host="127.0.0.1", port=8765)
    ipc.start()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
