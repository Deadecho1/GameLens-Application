import { durationToSeconds } from './duration';

export const MOVING_AVERAGE_WINDOW = 5;

/** Resolve run length in seconds (duration_seconds or HH:MM:SS string). */
export function getRunDurationSeconds(run) {
  if (!run || typeof run !== 'object') return 0;
  if (typeof run.duration_seconds === 'number' && Number.isFinite(run.duration_seconds)) {
    return run.duration_seconds;
  }
  return durationToSeconds(run.duration);
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
    return {
      ...point,
      movingAverage: sum / window.length,
    };
  });
}

/**
 * Build radar chart rows: chronological order, duration, and trailing moving average.
 */
export function buildRunDurationTrendSeries(runs = [], windowSize = MOVING_AVERAGE_WINDOW) {
  const ordered = sortRunsChronologically(runs);
  const base = ordered.map((run, index) => ({
    order: index + 1,
    durationSec: getRunDurationSeconds(run),
    runId: run.id,
    date: run.date,
    durationLabel: run.duration,
    run,
  }));
  return attachMovingAverage(base, windowSize);
}
