"""Metric dataclasses and aggregation for the pipeline benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

TIME_TOLERANCE = 5.0  # seconds


@dataclass
class ChoiceResult:
    index: int                    # 0-based position within the run
    options_match: bool
    selection_match: bool
    expected_options: list[str]
    actual_options: list[str]
    expected_selected: str
    actual_selected: str


@dataclass
class RunResult:
    run_index: int                # 1-based, from ground truth
    found: bool                   # whether the run JSON existed
    start_err: Optional[float]    # abs error in seconds, None if not found
    end_err: Optional[float]
    expected_choice_count: int
    actual_choice_count: int
    choices: list[ChoiceResult] = field(default_factory=list)

    @property
    def start_ok(self) -> bool:
        return self.start_err is not None and self.start_err <= TIME_TOLERANCE

    @property
    def end_ok(self) -> bool:
        return self.end_err is not None and self.end_err <= TIME_TOLERANCE

    @property
    def choice_count_ok(self) -> bool:
        return self.expected_choice_count == self.actual_choice_count


@dataclass
class VideoResult:
    video_stem: str
    expected_run_count: int
    actual_run_count: int
    runs: list[RunResult] = field(default_factory=list)

    @property
    def run_count_ok(self) -> bool:
        return self.expected_run_count == self.actual_run_count


@dataclass
class BenchmarkSummary:
    videos: list[VideoResult] = field(default_factory=list)

    # --- aggregated counts ------------------------------------------------

    def _all_runs(self) -> list[RunResult]:
        return [r for v in self.videos for r in v.runs]

    def _all_choices(self) -> list[ChoiceResult]:
        return [c for r in self._all_runs() for c in r.choices]

    def run_count_accuracy(self) -> tuple[int, int]:
        ok = sum(1 for v in self.videos if v.run_count_ok)
        return ok, len(self.videos)

    def start_time_accuracy(self) -> tuple[int, int]:
        runs = [r for r in self._all_runs() if r.found]
        ok = sum(1 for r in runs if r.start_ok)
        return ok, len(runs)

    def end_time_accuracy(self) -> tuple[int, int]:
        runs = [r for r in self._all_runs() if r.found]
        ok = sum(1 for r in runs if r.end_ok)
        return ok, len(runs)

    def choice_count_accuracy(self) -> tuple[int, int]:
        runs = [r for r in self._all_runs() if r.found]
        ok = sum(1 for r in runs if r.choice_count_ok)
        return ok, len(runs)

    def options_accuracy(self) -> tuple[int, int]:
        choices = self._all_choices()
        ok = sum(1 for c in choices if c.options_match)
        return ok, len(choices)

    def selection_accuracy(self) -> tuple[int, int]:
        choices = self._all_choices()
        ok = sum(1 for c in choices if c.selection_match)
        return ok, len(choices)

    def all_passed(self) -> bool:
        checks = [
            self.run_count_accuracy(),
            self.start_time_accuracy(),
            self.end_time_accuracy(),
            self.choice_count_accuracy(),
            self.options_accuracy(),
            self.selection_accuracy(),
        ]
        return all(ok == total for ok, total in checks if total > 0)

    # --- report -----------------------------------------------------------

    def print_report(self) -> None:
        print("\n=== Benchmark Results ===\n")

        for v in self.videos:
            count_mark = "✓" if v.run_count_ok else "✗"
            print(f"{v.video_stem}")
            print(
                f"  Runs: {v.expected_run_count} expected, "
                f"{v.actual_run_count} found  {count_mark}"
            )

            for r in v.runs:
                if not r.found:
                    print(f"  Run {r.run_index}: NOT FOUND ✗")
                    continue

                s_mark = "✓" if r.start_ok else "✗"
                e_mark = "✓" if r.end_ok else "✗"
                c_mark = "✓" if r.choice_count_ok else "✗"
                print(
                    f"  Run {r.run_index}: "
                    f"start_err={r.start_err:.2f}s {s_mark}  "
                    f"end_err={r.end_err:.2f}s {e_mark}"
                )
                print(
                    f"           choices: "
                    f"{r.actual_choice_count}/{r.expected_choice_count} found {c_mark}"
                )

                for i, c in enumerate(r.choices):
                    o_mark = "✓" if c.options_match else "✗"
                    s_mark2 = "✓" if c.selection_match else "✗"
                    print(
                        f"             choice {i + 1}: "
                        f"options={_match_label(c.options_match)} {o_mark}  "
                        f"selected={_match_label(c.selection_match)} {s_mark2}"
                    )
                    if not c.options_match:
                        print(f"               expected options: {c.expected_options}")
                        print(f"               actual   options: {c.actual_options}")
                    if not c.selection_match:
                        print(f"               expected selected: {c.expected_selected!r}")
                        print(f"               actual   selected: {c.actual_selected!r}")

            print()

        print("=== Aggregate ===")
        _print_metric("Run count accuracy",     self.run_count_accuracy())
        _print_metric(f"Start time within {TIME_TOLERANCE:.0f}s", self.start_time_accuracy())
        _print_metric(f"End time within {TIME_TOLERANCE:.0f}s",   self.end_time_accuracy())
        _print_metric("Choice count accuracy",  self.choice_count_accuracy())
        _print_metric("Options accuracy",        self.options_accuracy())
        _print_metric("Selection accuracy",      self.selection_accuracy())
        print()
        result = "PASS" if self.all_passed() else "FAIL"
        print(f"Overall: {result}")


def _match_label(ok: bool) -> str:
    return "MATCH" if ok else "MISMATCH"


def _print_metric(label: str, counts: tuple[int, int]) -> None:
    ok, total = counts
    if total == 0:
        print(f"  {label:<30} N/A")
    else:
        pct = 100.0 * ok / total
        print(f"  {label:<30} {ok}/{total} ({pct:.1f}%)")
