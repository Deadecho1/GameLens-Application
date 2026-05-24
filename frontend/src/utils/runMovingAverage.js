import { durationToSeconds } from './duration';

/** Trailing window for developer-facing session trend (current + prior runs). */
export const MOVING_AVERAGE_WINDOW = 10;

/** Resolve run length in seconds (duration_seconds or HH:MM:SS string). */
export function getRunDurationSeconds(run) {
  if (!run || typeof run !== 'object') return 0;
  if (typeof run.duration_seconds === 'number' && Number.isFinite(run.duration_seconds)) {
    return run.duration_seconds;
  }
  return durationToSeconds(run.duration);
}

/** Database / catalog run index (may have gaps); falls back to chronological slot. */
export function resolveRunIndex(run, chronologicalOrder) {
  if (run && typeof run.run_index === 'number' && Number.isFinite(run.run_index)) {
    return run.run_index;
  }
  if (run && typeof run.runIndex === 'number' && Number.isFinite(run.runIndex)) {
    return run.runIndex;
  }
  const fromId = Number.parseInt(String(run?.id ?? ''), 10);
  if (Number.isFinite(fromId)) return fromId;
  return chronologicalOrder;
}

/** Oldest → newest by `date`, then stable tie-break on `id`. */
export function sortRunsChronologically(runs = []) {
  return [...runs].sort((a, b) => {
    const ta = Date.parse(a?.date);
    const tb = Date.parse(b?.date);
    const aTime = Number.isFinite(ta) ? ta : 0;
    const bTime = Number.isFinite(tb) ? tb : 0;
    if (aTime !== bTime) return aTime - bTime;
    return String(a?.id ?? '').localeCompare(String(b?.id ?? ''));
  });
}

/**
 * For each chronologically ordered point, append `movingAverage` (window = current + prior runs).
 * @param {{ durationSec: number }[]} points — must already be in time order
 */
export function attachMovingAverage(points, windowSize = MOVING_AVERAGE_WINDOW) {
  const size = Math.max(1, windowSize);
  return points.map((point, index) => {
    const start = Math.max(0, index - size + 1);
    const window = points.slice(start, index + 1);
    const sum = window.reduce((acc, p) => acc + p.durationSec, 0);
    const trendValue = sum / window.length;
    return {
      ...point,
      movingAverage: trendValue,
      trendValue,
    };
  });
}

/**
 * Build radar chart rows: chronological order, duration, and trailing moving average.
 */
export function buildRunDurationTrendSeries(runs = [], windowSize = MOVING_AVERAGE_WINDOW) {
  const ordered = sortRunsChronologically(runs);
  const base = ordered.map((run, index) => ({
    run_index: resolveRunIndex(run, index + 1),
    order: index + 1,
    durationSec: getRunDurationSeconds(run),
    runId: run.id,
    date: run.date,
    durationLabel: run.duration,
    run,
  }));
  return attachMovingAverage(base, windowSize);
}
