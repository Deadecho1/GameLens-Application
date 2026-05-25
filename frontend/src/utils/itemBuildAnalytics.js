import { durationToSeconds } from './duration';

/** Global run average + per-item equipped run stats (Items BUILD tab). */
export function analyzeRunsForItems(runsHistory) {
  const runs = runsHistory ?? [];
  const globalAvgSeconds =
    runs.length > 0
      ? runs.reduce((acc, r) => acc + durationToSeconds(r.duration), 0) / runs.length
      : 0;

  const byItem = new Map();
  for (const run of runs) {
    const runSec = durationToSeconds(run.duration);
    const ids = new Set();
    for (const enc of run.bossEncounters ?? []) {
      for (const id of enc.loadout ?? []) ids.add(id);
    }
    for (const id of ids) {
      if (!byItem.has(id)) byItem.set(id, { totalSec: 0, runCount: 0 });
      const agg = byItem.get(id);
      agg.totalSec += runSec;
      agg.runCount += 1;
    }
  }

  const itemRunStats = new Map();
  for (const [id, agg] of byItem) {
    itemRunStats.set(id, {
      runCount: agg.runCount,
      avgEquippedRunSeconds:
        agg.runCount > 0 ? agg.totalSec / agg.runCount : globalAvgSeconds,
    });
  }

  return { globalAvgSeconds, itemRunStats };
}

export function formatMmSs(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}
