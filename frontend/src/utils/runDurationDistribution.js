import { durationToSeconds } from './duration';

/** Histogram buckets for run length (minutes). */
export const RUN_DURATION_BUCKET_DEFS = [
  { bucket: '0-5 min', minMinutes: 0, maxMinutes: 5 },
  { bucket: '5-10 min', minMinutes: 5, maxMinutes: 10 },
  { bucket: '10-15 min', minMinutes: 10, maxMinutes: 15 },
  { bucket: '15-20 min', minMinutes: 15, maxMinutes: 20 },
  { bucket: '20-25 min', minMinutes: 20, maxMinutes: 25 },
  { bucket: '25+ min', minMinutes: 25, maxMinutes: Infinity },
];

/**
 * Resolve run length in minutes from API shape (duration_seconds or HH:MM:SS string).
 */
export function getRunDurationMinutes(run) {
  if (!run || typeof run !== 'object') return 0;
  if (typeof run.duration_seconds === 'number' && Number.isFinite(run.duration_seconds)) {
    return run.duration_seconds / 60;
  }
  return durationToSeconds(run.duration) / 60;
}

/**
 * Group runs into fixed duration buckets for histogram display.
 * @returns {{ bucket: string, count: number, minMinutes: number, maxMinutes: number }[]}
 */
export function buildRunDurationDistribution(runs = []) {
  const chartData = RUN_DURATION_BUCKET_DEFS.map((def) => ({
    bucket: def.bucket,
    count: 0,
    minMinutes: def.minMinutes,
    maxMinutes: def.maxMinutes,
  }));

  for (const run of runs) {
    const minutes = getRunDurationMinutes(run);
    const row = chartData.find(
      (b) =>
        minutes >= b.minMinutes &&
        (b.maxMinutes === Infinity ? true : minutes < b.maxMinutes),
    );
    if (row) row.count += 1;
  }

  return chartData;
}
