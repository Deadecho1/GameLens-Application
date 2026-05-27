import { durationToSeconds, formatSecondsAsHMS } from './duration';

const EARLY_END_SEC = 5 * 60;
const MID_END_SEC = 15 * 60;

const PHASE_BUCKETS = [
  { id: 'early', label: 'Early Run', sublabel: '0 – 5 min', minSec: 0, maxSec: EARLY_END_SEC },
  { id: 'mid', label: 'Mid Run', sublabel: '5 – 15 min', minSec: EARLY_END_SEC, maxSec: MID_END_SEC },
  { id: 'late', label: 'Late Run', sublabel: '15+ min', minSec: MID_END_SEC, maxSec: Infinity },
];

/**
 * Collect per-run pick events for an item.
 * Uses run.itemPickups when present; otherwise infers first loadout appearance in bossEncounters.
 */
export function collectItemPickEvents(runsHistory) {
  const byItem = new Map();

  const addEvent = (itemId, run, pickedAtSec) => {
    const id = Number(itemId);
    if (!Number.isFinite(id)) return;
    if (!byItem.has(id)) byItem.set(id, []);
    byItem.get(id).push({
      runId: run.id,
      durationSec: durationToSeconds(run.duration),
      pickedAtSec: Math.max(0, pickedAtSec ?? 0),
    });
  };

  for (const run of runsHistory ?? []) {
    if (Array.isArray(run.itemPickups) && run.itemPickups.length > 0) {
      for (const pickup of run.itemPickups) {
        const itemId = pickup.itemId ?? pickup.item_id;
        const pickedAt =
          pickup.pickedAtSeconds ??
          pickup.picked_at_seconds ??
          pickup.pickedAt ??
          0;
        addEvent(itemId, run, pickedAt);
      }
      continue;
    }

    let cumulativeSec = 0;
    const seenInRun = new Set();
    for (const enc of run.bossEncounters ?? []) {
      const encSec = durationToSeconds(enc.lifespan);
      for (const itemId of enc.loadout ?? []) {
        const id = Number(itemId);
        if (seenInRun.has(id)) continue;
        seenInRun.add(id);
        addEvent(id, run, cumulativeSec);
      }
      cumulativeSec += encSec;
    }
  }

  return byItem;
}

/** First in-run pickup time (seconds) from itemPickups or earliest loadout appearance. */
export function getItemFirstAppearanceSeconds(run, itemId) {
  const id = Number(itemId);
  if (!Number.isFinite(id) || !run) return null;
  const events = collectItemPickEvents([run]).get(id);
  if (!events?.length) return null;
  return events[0].pickedAtSec;
}

export function getRunsContainingItem(runsHistory, itemId) {
  const events = collectItemPickEvents(runsHistory).get(Number(itemId)) ?? [];
  const runIds = new Set(events.map((e) => e.runId));
  return (runsHistory ?? []).filter((r) => runIds.has(r.id));
}

export function computeItemDetailAnalytics(catalog, runsHistory, itemId) {
  const id = Number(itemId);
  const item = catalog.find((i) => Number(i.id) === id);
  if (!item) return null;

  const pickEvents = collectItemPickEvents(runsHistory).get(id) ?? [];
  const runCount = pickEvents.length;

  const avgRunDurationSec =
    runCount > 0
      ? pickEvents.reduce((sum, e) => sum + e.durationSec, 0) / runCount
      : null;

  const synergyCounts = new Map();
  const runsWithItem = getRunsContainingItem(runsHistory, id);

  for (const run of runsWithItem) {
    const coItems = new Set();
    for (const enc of run.bossEncounters ?? []) {
      for (const otherId of enc.loadout ?? []) {
        const oid = Number(otherId);
        if (oid !== id && Number.isFinite(oid)) coItems.add(oid);
      }
    }
    if (Array.isArray(run.itemPickups)) {
      for (const p of run.itemPickups) {
        const oid = Number(p.itemId ?? p.item_id);
        if (oid !== id && Number.isFinite(oid)) coItems.add(oid);
      }
    }
    for (const oid of coItems) {
      synergyCounts.set(oid, (synergyCounts.get(oid) ?? 0) + 1);
    }
  }

  const topSynergies = [...synergyCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([synergyId, count]) => {
      const row = catalog.find((i) => Number(i.id) === synergyId);
      return {
        id: synergyId,
        name: row?.name ?? `Item ${synergyId}`,
        count,
        pct: runsWithItem.length > 0 ? Math.round((count / runsWithItem.length) * 100) : 0,
      };
    });

  const phaseCounts = { early: 0, mid: 0, late: 0 };
  for (const ev of pickEvents) {
    const t = ev.pickedAtSec;
    if (t < EARLY_END_SEC) phaseCounts.early += 1;
    else if (t < MID_END_SEC) phaseCounts.mid += 1;
    else phaseCounts.late += 1;
  }

  const pickPhaseChart = PHASE_BUCKETS.map((bucket) => ({
    ...bucket,
    count: phaseCounts[bucket.id] ?? 0,
  }));

  return {
    item,
    runCount,
    popularity: item.popularity ?? null,
    avgRunDurationSec,
    avgRunDurationLabel:
      avgRunDurationSec != null ? formatSecondsAsHMS(avgRunDurationSec) : '—',
    topSynergies,
    pickPhaseChart,
    totalPicks: pickEvents.length,
  };
}

export { PHASE_BUCKETS };
