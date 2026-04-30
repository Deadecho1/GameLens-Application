"""Shared GUI utilities and mixins."""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QGroupBox, QWidget


def populate_combo_restoring_selection(
    combo: QComboBox,
    items: List[str],
    preferred: str,
) -> None:
    """Clear and repopulate a combo box, restoring the previously selected item."""
    combo.blockSignals(True)
    combo.clear()
    for item in items:
        combo.addItem(item)
    combo.blockSignals(False)

    if items:
        index = combo.findText(preferred)
        combo.setCurrentIndex(index if index >= 0 else 0)


class ResponsiveFontMixin:
    """Mixin for QWidget subclasses that scale font size with window size.

    Usage:
        1. Call ``_setup_font_timer()`` in ``__init__``.
        2. Call ``_schedule_font_update()`` from ``resizeEvent``.
        3. Override ``_font_scale_params()`` to tune the ratios if needed.
    """

    def _setup_font_timer(self) -> None:
        self._font_timer: QTimer = QTimer(self)  # type: ignore[call-arg]
        self._font_timer.setSingleShot(True)
        self._font_timer.timeout.connect(self._apply_responsive_fonts)

    def _font_scale_params(self) -> tuple:
        """Return (min_size, max_size, min_width, min_height, width_ratio, height_ratio)."""
        return 14, 20, 900, 700, 78.0, 45.0

    def _schedule_font_update(self) -> None:
        self._font_timer.start(50)

    def _apply_font_recursive(self, widget: QWidget, point_size: int) -> None:
        font = QFont(widget.font())
        font.setPointSize(point_size)
        widget.setFont(font)
        for child in widget.findChildren(QWidget):
            child_font = QFont(child.font())
            child_font.setPointSize(point_size)
            child.setFont(child_font)

    def _apply_responsive_fonts(self) -> None:
        min_fs, max_fs, min_w, min_h, w_ratio, h_ratio = self._font_scale_params()
        width = max(self.width(), min_w)  # type: ignore[attr-defined]
        height = max(self.height(), min_h)  # type: ignore[attr-defined]
        point_size = max(min_fs, min(max_fs, int(min(width / w_ratio, height / h_ratio))))

        root_widget = (
            self.centralWidget()  # type: ignore[attr-defined]
            if hasattr(self, "centralWidget")
            else self
        )
        self._apply_font_recursive(root_widget, point_size)

        title_font = QFont(self.font())  # type: ignore[attr-defined]
        title_font.setPointSize(point_size + 1)
        for group_box in self.findChildren(QGroupBox):  # type: ignore[attr-defined]
            group_box.setFont(title_font)
