from typing import Dict, List

from .config import DetectorConfig
from .labels import LABEL_START, LABEL_END, LABEL_CHOICE, LABEL_DROP
from .models import DecodedRun, PeakEvent, RunCandidate


class RunDecoder:
    def __init__(self, config: DetectorConfig):
        self.config = config

    @staticmethod
    def events_inside_interval(events: List[PeakEvent], start_t: float, end_t: float) -> List[PeakEvent]:
        return [e for e in events if start_t <= e.time <= end_t]

    def short_run_penalty(self, duration: float) -> float:
        if duration >= self.config.min_run_seconds:
            return 0.0
        return (self.config.min_run_seconds - duration) / self.config.min_run_seconds

    def score_run_candidate(
        self,
        s: PeakEvent,
        e: PeakEvent,
        starts: List[PeakEvent],
        ends: List[PeakEvent],
        choices: List[PeakEvent],
        drops: List[PeakEvent],
    ) -> float:
        if e.time <= s.time:
            return float("-inf")

        duration = e.time - s.time
        inside_choices = self.events_inside_interval(choices, s.time, e.time)
        inside_drops = self.events_inside_interval(drops, s.time, e.time)

        outside_choices = [c for c in choices if c.time < s.time or c.time > e.time]
        outside_drops = [d for d in drops if d.time < s.time or d.time > e.time]

        inner_starts = [x for x in starts if s.time < x.time < e.time]
        inner_ends = [x for x in ends if s.time < x.time < e.time]

        inside_reward = sum(x.score for x in inside_choices + inside_drops)
        outside_penalty = sum(x.score for x in outside_choices + outside_drops)
        inner_boundary_penalty = sum(x.score for x in inner_starts + inner_ends)

        score = 0.0
        score += s.score
        score += e.score
        score += self.config.inside_event_reward_weight * inside_reward
        score -= self.config.short_run_penalty_weight * self.short_run_penalty(duration)
        score -= self.config.outside_event_penalty_weight * outside_penalty
        score -= self.config.inner_boundary_penalty_weight * inner_boundary_penalty

        return score

    def build_run_candidates(self, peaks: Dict[str, List[PeakEvent]]) -> List[RunCandidate]:
        starts = peaks[LABEL_START]
        ends = peaks[LABEL_END]
        choices = peaks[LABEL_CHOICE]
        drops = peaks[LABEL_DROP]

        candidates: List[RunCandidate] = []

        for s in starts:
            for e in ends:
                if e.time <= s.time:
                    continue

                score = self.score_run_candidate(s, e, starts, ends, choices, drops)
                if score == float("-inf"):
                    continue

                candidates.append(
                    RunCandidate(
                        start_event=s,
                        end_event=e,
                        start_time=s.time,
                        end_time=e.time,
                        score=score,
                    )
                )

        candidates.sort(key=lambda r: (r.end_time, r.start_time))
        return candidates

    def weighted_interval_scheduling(self, candidates: List[RunCandidate]) -> List[RunCandidate]:
        if not candidates:
            return []

        p = []
        for i in range(len(candidates)):
            j = i - 1
            while j >= 0 and candidates[j].end_time > candidates[i].start_time:
                j -= 1
            p.append(j)

        dp = [0.0] * len(candidates)
        take = [False] * len(candidates)

        for i in range(len(candidates)):
            include_score = candidates[i].score + (dp[p[i]] if p[i] >= 0 else 0.0)
            exclude_score = dp[i - 1] if i > 0 else 0.0

            if include_score > exclude_score and include_score > 0:
                dp[i] = include_score
                take[i] = True
            else:
                dp[i] = exclude_score
                take[i] = False

        selected: List[RunCandidate] = []
        i = len(candidates) - 1

        while i >= 0:
            include_score = candidates[i].score + (dp[p[i]] if p[i] >= 0 else 0.0)
            exclude_score = dp[i - 1] if i > 0 else 0.0

            if take[i] and include_score >= exclude_score and include_score > 0:
                selected.append(candidates[i])
                i = p[i]
            else:
                i -= 1

        selected.reverse()
        return selected

    def assign_middle_events_to_runs(
        self,
        selected_runs: List[RunCandidate],
        choice_events: List[PeakEvent],
        drop_events: List[PeakEvent],
    ) -> List[DecodedRun]:
        decoded: List[DecodedRun] = []

        for r in selected_runs:
            choices = [c for c in choice_events if r.start_time <= c.time <= r.end_time]
            drops = [d for d in drop_events if r.start_time <= d.time <= r.end_time]

            decoded.append(
                DecodedRun(
                    start=r.start_event,
                    end=r.end_event,
                    choices=choices,
                    drops=drops,
                )
            )

        return decoded

    def decode_runs(self, peaks: Dict[str, List[PeakEvent]]) -> List[DecodedRun]:
        run_candidates = self.build_run_candidates(peaks)
        selected_runs = self.weighted_interval_scheduling(run_candidates)
        return self.assign_middle_events_to_runs(
            selected_runs,
            peaks[LABEL_CHOICE],
            peaks[LABEL_DROP],
        )