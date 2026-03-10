from typing import Any, List, Tuple

from .choice_extract import HighlightDet, SelectionModel
from .pipeline import run_pipeline


class ChoiceExtractor:
    def __init__(self, selection_model: SelectionModel):
        self.selection_model = selection_model

    def process_frame(self, frame_path: str) -> Tuple[List[Any], HighlightDet | None]:
        # Run the pipeline to get options
        options = run_pipeline(frame_path)

        # Use the selection model to detect the highlight
        highlight_det = self.selection_model.detect_highlight(frame_path)

        return options, highlight_det
