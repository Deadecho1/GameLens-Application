from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .config import APP_NAME
from .models import ChoiceDetail, RunDetails


class RunDetailsDialog(QDialog):
    def __init__(self, details: RunDetails, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - {details.run_name}")
        self.resize(850, 720)
        self._build_ui(details)

    def _build_ui(self, details: RunDetails) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel(details.run_name)
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)

        duration_label = QLabel(f"Duration: {self._format_seconds(details.duration_seconds)}")

        root.addWidget(title)
        root.addWidget(duration_label)

        # Acquired Items
        items_group = QGroupBox(f"Acquired Items ({len(details.selected_items)})")
        items_layout = QVBoxLayout(items_group)
        if details.selected_items:
            items_list = QListWidget()
            for item in details.selected_items:
                items_list.addItem(item)
            items_list.setMaximumHeight(130)
            items_layout.addWidget(items_list)
        else:
            items_layout.addWidget(QLabel("No items recorded."))
        root.addWidget(items_group)

        # Choices
        choices_group = QGroupBox(f"Choices ({len(details.choices)})")
        choices_inner = QVBoxLayout(choices_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        if details.choices:
            for i, choice in enumerate(details.choices, start=1):
                card = self._make_choice_card(i, choice)
                scroll_layout.addWidget(card)
            scroll_layout.addStretch(1)
        else:
            scroll_layout.addWidget(QLabel("No choices recorded."))

        scroll.setWidget(scroll_content)
        choices_inner.addWidget(scroll)
        root.addWidget(choices_group, 1)

    def _make_choice_card(self, index: int, choice: ChoiceDetail) -> QGroupBox:
        box = QGroupBox(f"Choice {index}")
        layout = QVBoxLayout(box)

        if choice.options:
            for option in choice.options:
                is_selected = option == choice.selected
                marker = "✓" if is_selected else "•"
                lbl = QLabel(f"  {marker}  {option}")
                if is_selected:
                    lbl.setStyleSheet("color: #2ecc71; font-weight: bold;")
                layout.addWidget(lbl)
        else:
            selected_label = QLabel(f"  ✓  {choice.selected}")
            selected_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            layout.addWidget(selected_label)

        return box

    def _format_seconds(self, seconds: float) -> str:
        total = int(seconds)
        minutes, sec = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {sec}s"
        if minutes > 0:
            return f"{minutes}m {sec}s"
        return f"{sec}s"
