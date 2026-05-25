import { durationToSeconds } from './duration';

export function collectGlobalLifespanSeconds(bossId, dashboard) {
  const bosses = dashboard?.bosses ?? [];
  const runsHistory = dashboard?.runsHistory ?? [];
  const boss = bosses.find((b) => b.id === bossId);
  const out = [];

  for (const run of runsHistory) {
    const enc = run.bossEncounters;
    if (!Array.isArray(enc)) continue;
    for (const e of enc) {
      if (e.bossId !== bossId || !e.lifespan) continue;
      const s = durationToSeconds(e.lifespan);
      if (s > 0) out.push(s);
    }
  }

  const extra = boss?.globalLifespanSamples;
  if (Array.isArray(extra)) {
    for (const raw of extra) {
      const s = durationToSeconds(raw);
      if (s > 0) out.push(s);
    }
  }

  if (out.length === 0 && boss?.lifespan) {
    const s = durationToSeconds(boss.lifespan);
    if (s > 0) out.push(s);
  }

  return out;
}

export function meanSeconds(values) {
  if (!values.length) return 0;
  return values.reduce((a, s) => a + s, 0) / values.length;
}

export function computeBossGlobalMetrics(bossId, dashboard) {
  const seconds = collectGlobalLifespanSeconds(bossId, dashboard);
  return {
    globalAvgSec: Math.round(meanSeconds(seconds)),
    encounterCount: seconds.length,
  };
}
