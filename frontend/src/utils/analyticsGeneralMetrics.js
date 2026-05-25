import { durationToSeconds } from './duration';

/** Aggregate GENERAL-tab KPIs from dashboard slices (version-scoped data). */
export function computeGeneralMetrics(data) {
  const runsHistory = data?.dashboard?.runsHistory ?? [];
  const bosses = data?.dashboard?.bosses ?? [];
  const items = data?.dashboard?.items ?? [];

  const n = runsHistory.length;
  let avgSec = 0;
  let longestSec = 0;
  if (n > 0) {
    let sum = 0;
    for (const run of runsHistory) {
      const sec = durationToSeconds(run.duration);
      sum += sec;
      if (sec > longestSec) longestSec = sec;
    }
    avgSec = sum / n;
  }

  const total = bosses.length;
  let defeated = 0;
  for (const b of bosses) {
    if (String(b.status ?? '').toLowerCase() === 'defeated') defeated += 1;
  }
  const bossKillPercent = total > 0 ? Math.round((defeated / total) * 100) : 0;

  let mostPopularPopularity = null;
  if (items.length > 0) {
    const best = items.reduce((a, it) =>
      (it.popularity ?? 0) > (a.popularity ?? 0) ? it : a,
    );
    mostPopularPopularity =
      best.popularity != null ? Math.round(Number(best.popularity)) : null;
  }

  return {
    totalRuns: n,
    avgSec,
    longestSec,
    totalItems: items.length,
    bossKillPercent,
    mostPopularPopularity,
  };
}
