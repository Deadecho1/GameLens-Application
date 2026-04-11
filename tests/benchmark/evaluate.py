"""Core comparison logic for the pipeline benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import BenchmarkSummary, ChoiceResult, RunResult, VideoResult


def load_ground_truth(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalise(text: str) -> str:
    return text.strip().lower()


def _options_set(options: list[str]) -> set[str]:
    return {_normalise(o) for o in options}


def _compare_choices(
    expected_choices: list[dict],
    actual_choices: list[dict],
) -> list[ChoiceResult]:
    results: list[ChoiceResult] = []
    # Pair by position; extra actual choices are ignored, missing ones are not scored
    for i, exp in enumerate(expected_choices):
        exp_options = exp.get("options", [])
        exp_selected = exp.get("selected_option", "")

        if i < len(actual_choices):
            act = actual_choices[i]
            act_options = act.get("options", [])
            act_selected = act.get("selected_option", "")
        else:
            act_options = []
            act_selected = ""

        options_match = _options_set(exp_options) == _options_set(act_options)
        selection_match = _normalise(exp_selected) == _normalise(act_selected)

        results.append(
            ChoiceResult(
                index=i,
                options_match=options_match,
                selection_match=selection_match,
                expected_options=exp_options,
                actual_options=act_options,
                expected_selected=exp_selected,
                actual_selected=act_selected,
            )
        )
    return results


def _load_run_json(run_json_dir: Path, video_stem: str, run_index: int) -> dict | None:
    filename = f"{video_stem}_run_{run_index}.json"
    path = run_json_dir / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_video(
    video_stem: str,
    expected: dict,
    run_json_dir: Path,
) -> VideoResult:
    expected_runs: list[dict] = expected.get("runs", [])

    # Count actual run files for this video
    actual_files = sorted(run_json_dir.glob(f"{video_stem}_run_*.json"))
    actual_run_count = len(actual_files)

    video_result = VideoResult(
        video_stem=video_stem,
        expected_run_count=len(expected_runs),
        actual_run_count=actual_run_count,
    )

    for exp_run in expected_runs:
        run_index: int = exp_run["run_index"]
        actual = _load_run_json(run_json_dir, video_stem, run_index)

        if actual is None:
            video_result.runs.append(
                RunResult(
                    run_index=run_index,
                    found=False,
                    start_err=None,
                    end_err=None,
                    expected_choice_count=len(exp_run.get("choices", [])),
                    actual_choice_count=0,
                )
            )
            continue

        start_err = abs(actual["start_time"] - exp_run["start_time"])
        end_err = abs(actual["end_time"] - exp_run["end_time"])

        exp_choices = exp_run.get("choices", [])
        act_choices = actual.get("choices", [])
        choice_results = _compare_choices(exp_choices, act_choices)

        video_result.runs.append(
            RunResult(
                run_index=run_index,
                found=True,
                start_err=start_err,
                end_err=end_err,
                expected_choice_count=len(exp_choices),
                actual_choice_count=len(act_choices),
                choices=choice_results,
            )
        )

    return video_result


def evaluate_all(
    ground_truth: dict,
    run_json_dir: Path,
) -> BenchmarkSummary:
    summary = BenchmarkSummary()
    for video_stem, expected in ground_truth.items():
        summary.videos.append(evaluate_video(video_stem, expected, run_json_dir))
    return summary
