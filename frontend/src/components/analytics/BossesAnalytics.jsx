import { useEffect, useId, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Award,
  ChartColumn,
  Crown,
  Globe,
  Radar,
  RotateCcw,
  Search,
  Swords,
  Timer,
  Zap,
} from 'lucide-react';
import { durationToSeconds, formatSecondsAsHMS, secondsToMinutes } from '../../utils/duration';
import { bossAccentDotStyle, itemAccentDotStyle } from './items/itemUi';
import { computeBossGlobalMetrics } from '../../utils/analyticsBossMetrics';
import DeltaIndicator from './DeltaIndicator';

/** Seconds from all analyzed sessions (runsHistory.bossEncounters) plus boss.globalLifespanSamples; fallback: single lifespan. */
function collectGlobalLifespanSeconds(bossId, dashboard) {
  const bosses = dashboard.bosses ?? [];
  const runsHistory = dashboard.runsHistory ?? [];
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

function meanSeconds(values) {
  if (!values.length) return 0;
  return values.reduce((a, s) => a + s, 0) / values.length;
}

const SIM_SLOT_COUNT = 5;
const EMPTY_SIM_SLOTS = Array.from({ length: SIM_SLOT_COUNT }, () => null);

const HISTOGRAM_BRACKETS = [
  { bracket: '0-1m', min: 0, max: 60 },
  { bracket: '1-2m', min: 60, max: 120 },
  { bracket: '2-5m', min: 120, max: 300 },
  { bracket: '5m+', min: 300, max: Number.POSITIVE_INFINITY },
];

function lifespanHistogramData(secondsList) {
  const counts = [0, 0, 0, 0];
  for (const s of secondsList) {
    if (s < 60) counts[0]++;
    else if (s < 120) counts[1]++;
    else if (s < 300) counts[2]++;
    else counts[3]++;
  }
  return HISTOGRAM_BRACKETS.map((b, i) => ({
    bracket: b.bracket,
    count: counts[i],
  }));
}

/** Longer mean lifespan → better rank (rank 1 = longest average fight time). */
function fightTimeRanking(bossId, dashboard) {
  const bosses = dashboard.bosses ?? [];
  if (!bosses.length) return { rank: null, total: 0 };

  const scored = bosses.map((b) => ({
    id: b.id,
    mean: meanSeconds(collectGlobalLifespanSeconds(b.id, dashboard)),
  }));
  const sorted = [...scored].sort((a, b) => b.mean - a.mean);
  const rank = sorted.findIndex((x) => x.id === bossId) + 1;
  return { rank, total: bosses.length };
}

function normalizeLoadoutIds(loadout, validIds) {
  const allowed = new Set(validIds);
  return (Array.isArray(loadout) ? loadout : [])
    .filter((id) => allowed.has(id))
    .slice(0, SIM_SLOT_COUNT)
    .sort((a, b) => a - b);
}

/**
 * Historical encounters with loadout tags: lowest mean lifespan wins.
 * Falls back to boss.itemEffectiveness (strongest catalog row) if no tagged runs exist.
 */
function computeMostLethalSynergy(bossId, dashboard) {
  const items = dashboard.items ?? [];
  const validIds = items.map((i) => i.id);
  const runsHistory = dashboard.runsHistory ?? [];
  const grouped = new Map();

  for (const run of runsHistory) {
    for (const enc of run.bossEncounters || []) {
      if (enc.bossId !== bossId || !enc.lifespan) continue;
      const ids = normalizeLoadoutIds(enc.loadout, validIds);
      if (!ids.length) continue;
      const sec = durationToSeconds(enc.lifespan);
      if (sec <= 0) continue;
      const key = ids.join(':');
      if (!grouped.has(key)) grouped.set(key, { itemIds: [...ids], secs: [] });
      grouped.get(key).secs.push(sec);
    }
  }

  const allSeconds = collectGlobalLifespanSeconds(bossId, dashboard);
  const globalMean = meanSeconds(allSeconds);
  const totalEncounters = allSeconds.length;

  if (grouped.size === 0) {
    const boss = dashboard.bosses?.find((b) => b.id === bossId);
    const rows = boss?.itemEffectiveness ?? [];
    if (!rows.length || totalEncounters === 0 || globalMean <= 0) return null;
    const bestRow = rows.reduce((a, b) =>
      (Number(b.timeReductionVsGlobalPct) || 0) > (Number(a.timeReductionVsGlobalPct) || 0) ? b : a
    );
    const pct = Math.max(0, Math.round(Number(bestRow.timeReductionVsGlobalPct) || 0));
    return {
      itemIds: [bestRow.itemId],
      impactEfficiencyPct: pct,
      comboSampleCount: totalEncounters,
      totalEncounters,
      meanSecForCombo: Math.round(globalMean * (1 - pct / 100)),
      source: 'catalog',
    };
  }

  let best = null;
  for (const [, { itemIds, secs }] of grouped) {
    const m = meanSeconds(secs);
    if (!best || m < best.meanSec) {
      best = { itemIds, meanSec: m, comboSampleCount: secs.length };
    }
  }
  if (!best) return null;

  const impactEfficiencyPct =
    globalMean > 0
      ? Math.max(0, Math.min(99, Math.round((1 - best.meanSec / globalMean) * 100)))
      : 0;

  return {
    itemIds: best.itemIds,
    impactEfficiencyPct,
    comboSampleCount: best.comboSampleCount,
    totalEncounters,
    meanSecForCombo: Math.round(best.meanSec),
    source: 'historical',
  };
}

function estimateSimulatorFightTime(bossId, dashboard, equippedIds, globalAvgSec) {
  if (!bossId || globalAvgSec <= 0 || equippedIds.length === 0) {
    return { projectedSec: globalAvgSec, hasEstimate: false, usedExactMatch: false };
  }

  const items = dashboard.items ?? [];
  const validIds = items.map((i) => i.id);
  const runsHistory = dashboard.runsHistory ?? [];
  const normalizedEquipped = [...equippedIds].sort((a, b) => a - b);
  const exactKey = normalizedEquipped.join(':');
  const encounters = [];

  for (const run of runsHistory) {
    for (const enc of run.bossEncounters || []) {
      if (enc.bossId !== bossId || !enc.lifespan) continue;
      const sec = durationToSeconds(enc.lifespan);
      if (sec <= 0) continue;
      const ids = normalizeLoadoutIds(enc.loadout, validIds);
      encounters.push({ sec, ids });
    }
  }

  const exactSecs = encounters
    .filter((row) => row.ids.join(':') === exactKey)
    .map((row) => row.sec);
  if (exactSecs.length > 0) {
    return {
      projectedSec: Math.max(15, Math.round(meanSeconds(exactSecs))),
      hasEstimate: true,
      usedExactMatch: true,
    };
  }

  let sumDeltaSec = 0;
  let itemsWithHistory = 0;
  for (const itemId of normalizedEquipped) {
    const secsForItem = encounters
      .filter((row) => row.ids.includes(itemId))
      .map((row) => row.sec);
    if (secsForItem.length === 0) continue;
    const itemMeanSec = meanSeconds(secsForItem);
    sumDeltaSec += globalAvgSec - itemMeanSec;
    itemsWithHistory += 1;
  }

  if (itemsWithHistory === 0) {
    return { projectedSec: globalAvgSec, hasEstimate: false, usedExactMatch: false };
  }

  const aggregateDeltaSec = sumDeltaSec / normalizedEquipped.length;
  return {
    projectedSec: Math.max(15, Math.round(globalAvgSec - aggregateDeltaSec)),
    hasEstimate: true,
    usedExactMatch: false,
  };
}

/**
 * BOSSES — Master-detail tactical intel. dashboard.bosses + runsHistory + dashboard.items (gear).
 */
export default function BossesAnalytics({ data, compareBaseline = null }) {
  const dashboard = data.dashboard;
  const baselineDashboard = compareBaseline?.dashboard;
  const bosses = dashboard.bosses ?? [];
  const itemsCatalog = dashboard.items ?? [];
  const [selectedBossId, setSelectedBossId] = useState(null);
  /** Item combination simulator: up to 5 equipped item ids. */
  const [simSlots, setSimSlots] = useState(() => [...EMPTY_SIM_SLOTS]);
  const [libraryQuery, setLibraryQuery] = useState('');
  const chartGradId = useId().replace(/:/g, '');
  const simCompareChartId = useId().replace(/:/g, '');

  useEffect(() => {
    if (!bosses.length) {
      setSelectedBossId(null);
      return;
    }
    setSelectedBossId((prev) => {
      if (prev != null && bosses.some((b) => b.id === prev)) return prev;
      return bosses[0].id;
    });
  }, [bosses]);

  useEffect(() => {
    setSimSlots([...EMPTY_SIM_SLOTS]);
    setLibraryQuery('');
  }, [selectedBossId]);

  const selected = useMemo(
    () => bosses.find((b) => b.id === selectedBossId) ?? null,
    [bosses, selectedBossId]
  );

  const globalSeconds = useMemo(
    () => (selected ? collectGlobalLifespanSeconds(selected.id, dashboard) : []),
    [selected, dashboard]
  );

  const globalAvgSec = useMemo(() => Math.round(meanSeconds(globalSeconds)), [globalSeconds]);
  const globalAvgLabel = formatSecondsAsHMS(globalAvgSec);
  const globalEncounterCount = globalSeconds.length;

  const histogramData = useMemo(() => lifespanHistogramData(globalSeconds), [globalSeconds]);

  const survivalRank = useMemo(
    () =>
      selected
        ? fightTimeRanking(selected.id, dashboard)
        : { rank: null, total: 0 },
    [selected, dashboard]
  );

  const mostLethal = useMemo(
    () => (selected ? computeMostLethalSynergy(selected.id, dashboard) : null),
    [selected, dashboard]
  );

  const baselineBossMetrics = useMemo(() => {
    if (!baselineDashboard || !selected) return null;
    const exists = (baselineDashboard.bosses ?? []).some((b) => b.id === selected.id);
    if (!exists) return null;
    return computeBossGlobalMetrics(selected.id, baselineDashboard);
  }, [baselineDashboard, selected]);

  const baselineMostLethal = useMemo(() => {
    if (!baselineDashboard || !selected) return null;
    return computeMostLethalSynergy(selected.id, baselineDashboard);
  }, [baselineDashboard, selected]);

  const equippedIds = useMemo(() => simSlots.filter((id) => id != null), [simSlots]);
  const itemImpactEstimate = useMemo(
    () =>
      selected
        ? estimateSimulatorFightTime(selected.id, dashboard, equippedIds, globalAvgSec)
        : { projectedSec: globalAvgSec, hasEstimate: false, usedExactMatch: false },
    [selected, dashboard, equippedIds, globalAvgSec]
  );
  const synergyProjectedSec = itemImpactEstimate.projectedSec;
  const synergyFaster = itemImpactEstimate.hasEstimate && synergyProjectedSec < globalAvgSec;
  const synergyDeltaPct = useMemo(() => {
    if (!itemImpactEstimate.hasEstimate || globalAvgSec <= 0) return 0;
    return Math.round((1 - synergyProjectedSec / globalAvgSec) * 100);
  }, [itemImpactEstimate.hasEstimate, globalAvgSec, synergyProjectedSec]);
  const compareBarData = useMemo(() => {
    const baseFill = '#575f6b';
    const synergyFill = equippedIds.length === 0 ? '#64748b' : synergyFaster ? '#22d3ee' : '#fb923c';
    return [
      {
        key: 'base',
        label: 'Average fight time',
        seconds: globalAvgSec,
        minutes: secondsToMinutes(globalAvgSec),
        fill: baseFill,
      },
      {
        key: 'synergy',
        label: 'Estimated time',
        seconds: synergyProjectedSec,
        minutes: secondsToMinutes(synergyProjectedSec),
        fill: synergyFill,
      },
    ];
  }, [globalAvgSec, synergyProjectedSec, equippedIds.length, synergyFaster]);

  const toggleSimItem = (itemId) => {
    setSimSlots((slots) => {
      const idxFound = slots.findIndex((s) => s === itemId);
      if (idxFound !== -1) {
        const next = [...slots];
        next[idxFound] = null;
        return next;
      }
      const emptyIdx = slots.findIndex((s) => s === null);
      if (emptyIdx === -1) return slots;
      const next = [...slots];
      next[emptyIdx] = itemId;
      return next;
    });
  };

  const clearSimLoadout = () => setSimSlots([...EMPTY_SIM_SLOTS]);

  const filteredLibraryItems = useMemo(() => {
    const q = libraryQuery.trim().toLowerCase();
    if (!q) return itemsCatalog;
    return itemsCatalog.filter((item) => String(item.name ?? '').toLowerCase().includes(q));
  }, [itemsCatalog, libraryQuery]);

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <Swords className="h-5 w-5 text-cyan-400" strokeWidth={1.25} aria-hidden />
          <h3 className="font-display text-xl font-bold uppercase tracking-[0.12em] text-slate-100 md:text-2xl">
            BOSS ANALYTICS
          </h3>
        </div>
      </header>

      <div className="flex min-h-[min(640px,75vh)] flex-col gap-4 lg:flex-row lg:gap-0 lg:rounded-2xl lg:border lg:border-slate-800/90 lg:bg-slate-950/40 lg:shadow-[inset_0_1px_0_rgba(34,211,238,0.06)]">
        <aside className="flex w-full flex-col border-slate-800/80 lg:w-[25%] lg:min-w-[220px] lg:max-w-[320px] lg:border-r lg:bg-slate-950/50">
          <div className="flex items-center gap-2 border-b border-slate-800/80 px-4 py-3">
            <Radar className="h-4 w-4 text-cyan-400/90" strokeWidth={1.25} aria-hidden />
            <span className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
              Select a boss
            </span>
          </div>
          <nav
            className="flex flex-1 flex-col gap-1.5 overflow-y-auto p-3 [scrollbar-color:rgba(51,65,85,0.8)_transparent]"
            aria-label="Boss list"
          >
            {bosses.length === 0 ? (
              <p className="font-data px-2 py-6 text-center text-sm text-slate-500">No bosses listed.</p>
            ) : (
              bosses.map((boss) => {
                const active = boss.id === selectedBossId;
                return (
                  <motion.button
                    key={boss.id}
                    type="button"
                    onClick={() => setSelectedBossId(boss.id)}
                    whileHover={{ x: 4 }}
                    transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                    className={`group relative flex w-full items-center gap-3 rounded-xl border px-3 py-3 text-left transition-[filter,box-shadow,border-color,background-color] ${
                      active
                        ? 'border-cyan-400/45 bg-cyan-500/12 shadow-[0_0_24px_rgba(34,211,238,0.28),inset_0_0_20px_rgba(34,211,238,0.06)] brightness-110'
                        : 'border-slate-700/90 bg-slate-900/40 brightness-100 hover:border-slate-600 hover:bg-slate-900/70 hover:brightness-125'
                    } `}
                  >
                    {active && (
                      <motion.span
                        layoutId="boss-rail-glow"
                        className="pointer-events-none absolute inset-0 rounded-xl ring-1 ring-cyan-400/35"
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      />
                    )}
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={bossAccentDotStyle(boss.id, active)}
                      aria-hidden
                    />
                    <span
                      className={`font-display text-xs font-bold uppercase tracking-[0.14em] ${
                        active ? 'text-cyan-100' : 'text-slate-400 group-hover:text-slate-200'
                      }`}
                    >
                      {boss.name}
                    </span>
                  </motion.button>
                );
              })
            )}
          </nav>
        </aside>

        <section className="relative flex min-h-[480px] flex-1 flex-col lg:w-[75%]">
          <div
            className="gl-terminal-scanlines pointer-events-none absolute inset-0 rounded-none opacity-70 lg:rounded-r-2xl"
            aria-hidden
          />
          <div className="relative flex flex-1 flex-col p-4 backdrop-blur-sm md:p-6 lg:p-8">
            <AnimatePresence mode="wait">
              {!selected ? (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="flex flex-1 items-center justify-center"
                >
                  <p className="font-data text-sm text-slate-500">Select a boss from the list.</p>
                </motion.div>
              ) : (
                <motion.div
                  key={selected.id}
                  initial={{ opacity: 0, x: 18 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -14 }}
                  transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
                  className="flex flex-1 flex-col gap-8"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                      <h2 className="font-display text-3xl font-black uppercase tracking-[0.18em] text-white [text-shadow:0_0_40px_rgba(34,211,238,0.25)] md:text-4xl lg:text-5xl">
                        {selected.name}
                      </h2>
                      <p className="font-data mt-2 text-xs uppercase tracking-[0.25em] text-cyan-500/70">
                        Global performance metrics
                      </p>
                    </div>
                    {survivalRank.rank != null && survivalRank.total > 0 ? (
                      <div className="flex items-center gap-3 rounded-xl border border-slate-700/90 bg-slate-950/55 px-4 py-3 ring-1 ring-cyan-500/10 backdrop-blur-md">
                        <Award className="h-8 w-8 shrink-0 text-cyan-400/90" strokeWidth={1.15} aria-hidden />
                        <div>
                          <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                            Fight time ranking
                          </p>
                          <p className="font-data text-xl font-bold tabular-nums tracking-tight text-cyan-100 md:text-2xl">
                            #{survivalRank.rank} of {survivalRank.total}{' '}
                            {survivalRank.total === 1 ? 'Boss' : 'Bosses'}
                          </p>
                          <p className="font-data mt-1 text-[10px] text-slate-500">
                            Ranked by average fight time
                          </p>
                        </div>
                      </div>
                    ) : null}
                  </div>

                  <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
                    <div className="rounded-2xl border border-cyan-500/20 bg-slate-900/50 p-6 shadow-[0_0_40px_rgba(34,211,238,0.08),inset_0_1px_0_rgba(255,255,255,0.04)] ring-1 ring-cyan-500/10 backdrop-blur-md">
                      <div className="mb-4 flex items-center gap-2">
                        <Timer className="h-5 w-5 text-cyan-300" strokeWidth={1.25} aria-hidden />
                        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-200/90">
                          Average fight time
                        </h3>
                      </div>
                      <p className="font-data text-[10px] uppercase tracking-wider text-slate-500">
                        Across all recorded boss fights
                      </p>
                      <p
                        className="mt-4 font-data text-4xl font-bold tabular-nums tracking-tight text-cyan-100 md:text-5xl lg:text-6xl"
                        style={{ textShadow: '0 0 32px rgba(34, 211, 238, 0.35)' }}
                      >
                        {globalAvgLabel}
                      </p>
                      {baselineBossMetrics ? (
                        <DeltaIndicator
                          kind="duration"
                          baseline={baselineBossMetrics.globalAvgSec}
                          current={globalAvgSec}
                        />
                      ) : null}
                      <p className="font-data mt-2 text-xs tabular-nums text-slate-500">
                        {globalAvgSec > 0 ? `${secondsToMinutes(globalAvgSec)} min mean` : '—'}
                      </p>
                    </div>

                    <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-3 backdrop-blur-md md:flex-col md:items-stretch md:px-5 md:py-4">
                      <div className="flex items-center gap-2 text-slate-500">
                        <Globe className="h-4 w-4 shrink-0 text-cyan-500/80" strokeWidth={1.25} aria-hidden />
                        <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                          Global
                        </span>
                      </div>
                      <p className="font-data text-2xl font-bold tabular-nums text-white md:text-3xl">
                        {globalEncounterCount}
                      </p>
                      {baselineBossMetrics ? (
                        <DeltaIndicator
                          kind="count"
                          baseline={baselineBossMetrics.encounterCount}
                          current={globalEncounterCount}
                        />
                      ) : null}
                      <p className="font-data text-[10px] uppercase tracking-wider text-slate-600">
                        Total boss fights
                      </p>
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 rounded-2xl border border-slate-800/90 bg-slate-950/35 p-4 backdrop-blur-md md:p-5">
                    <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <ChartColumn className="h-4 w-4 text-cyan-400/90" strokeWidth={1.25} aria-hidden />
                        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-white/90">
                          Fight time distribution
                        </h3>
                      </div>
                      <span className="font-data text-[10px] text-slate-500">Fight length · number of fights</span>
                    </div>
                    <p className="font-data mb-4 text-[10px] text-slate-600">
                      Boss fights grouped by length — shows time consistency.
                    </p>
                    <div className="h-[280px] w-full min-w-0 md:h-[300px]">
                      {globalEncounterCount === 0 ? (
                        <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-800">
                          <p className="font-data text-sm text-slate-500">No global samples for this boss.</p>
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={histogramData} margin={{ top: 8, right: 8, left: 4, bottom: 8 }}>
                            <defs>
                              <linearGradient id={chartGradId} x1="0" y1="0" x2="1" y2="0">
                                <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.92} />
                                <stop offset="100%" stopColor="#7c3aed" stopOpacity={0.88} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid
                              strokeDasharray="3 6"
                              stroke="#334155"
                              strokeOpacity={0.55}
                              vertical={false}
                            />
                            <XAxis
                              dataKey="bracket"
                              tick={{ fill: '#e2e8f0', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                              axisLine={{ stroke: '#475569' }}
                              tickLine={{ stroke: '#475569' }}
                            />
                            <YAxis
                              allowDecimals={false}
                              tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                              axisLine={{ stroke: '#475569' }}
                              tickLine={{ stroke: '#475569' }}
                              label={{
                                value: 'Number of fights',
                                angle: -90,
                                position: 'insideLeft',
                                fill: '#64748b',
                                fontSize: 10,
                                fontFamily: 'JetBrains Mono, monospace',
                              }}
                            />
                            <Tooltip
                              cursor={{ fill: 'rgba(51, 65, 85, 0.2)' }}
                              content={({ active, payload }) => {
                                if (!active || !payload?.length) return null;
                                const row = payload[0].payload;
                                return (
                                  <div className="rounded-lg border border-cyan-900/50 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                                    <p className="font-data text-[10px] font-semibold uppercase tracking-wider text-cyan-400/90">
                                      {row.bracket}
                                    </p>
                                    <p className="font-data mt-1 tabular-nums text-sm text-white">
                                      {row.count} fight{row.count === 1 ? '' : 's'}
                                    </p>
                                  </div>
                                );
                              }}
                            />
                            <Bar
                              dataKey="count"
                              name="Number of fights"
                              fill={`url(#${chartGradId})`}
                              radius={[6, 6, 0, 0]}
                              maxBarSize={72}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </div>

                  <div className="space-y-6">
                    <motion.section
                      initial={false}
                      animate={{
                        boxShadow: [
                          '0 0 24px rgba(251,191,36,0.14), 0 0 52px rgba(168,85,247,0.1)',
                          '0 0 36px rgba(168,85,247,0.2), 0 0 60px rgba(251,191,36,0.12)',
                          '0 0 24px rgba(251,191,36,0.14), 0 0 52px rgba(168,85,247,0.1)',
                        ],
                      }}
                      transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
                      className="relative overflow-hidden rounded-2xl border-2 border-amber-400/45 bg-slate-950/80 p-5 ring-2 ring-purple-500/30 backdrop-blur-md md:p-6"
                    >
                      <motion.div
                        className="pointer-events-none absolute inset-y-0 -left-1/3 w-2/3 opacity-25"
                        style={{
                          background:
                            'linear-gradient(90deg, transparent, rgba(251,191,36,0.35), rgba(168,85,247,0.2), transparent)',
                        }}
                        animate={{ x: ['-20%', '120%'] }}
                        transition={{ duration: 3.8, repeat: Infinity, ease: 'linear' }}
                      />
                      <div className="relative">
                        <div className="mb-4 flex flex-wrap items-center gap-3">
                          <Crown
                            className="h-7 w-7 text-amber-300 drop-shadow-[0_0_14px_rgba(251,191,36,0.55)]"
                            strokeWidth={1.2}
                            aria-hidden
                          />
                          <div>
                            <p className="font-display text-[10px] font-bold uppercase tracking-[0.28em] text-amber-200/80">
                              Best items
                            </p>
                            <h3 className="font-display text-lg font-black uppercase tracking-[0.14em] text-transparent bg-linear-to-r from-amber-200 via-fuchsia-200 to-amber-100 bg-clip-text md:text-xl">
                              Best item combination
                            </h3>
                          </div>
                        </div>
                        {!mostLethal ? (
                          <p className="font-data text-sm text-slate-500">
                            Not enough aggregate data to rank loadouts for this target.
                          </p>
                        ) : (
                          <>
                            <div className="mb-4 flex flex-wrap gap-2">
                              {mostLethal.itemIds.map((id) => {
                                const item = itemsCatalog.find((i) => i.id === id);
                                if (!item) return null;
                                return (
                                  <span
                                    key={id}
                                    className="inline-flex items-center gap-2 rounded-lg border border-amber-300/45 bg-slate-900/90 px-3 py-2 shadow-[0_0_18px_rgba(251,191,36,0.2)]"
                                  >
                                    <span
                                      className="h-2 w-2 shrink-0 rounded-full"
                                      style={itemAccentDotStyle(item.id, true)}
                                      aria-hidden
                                    />
                                    <span className="font-display text-[10px] font-bold uppercase tracking-wide text-amber-100">
                                      {item.name}
                                    </span>
                                  </span>
                                );
                              })}
                            </div>
                            <p className="font-display text-xs font-bold uppercase tracking-wider text-slate-200 md:text-sm">
                              {mostLethal.itemIds
                                .map((id) => itemsCatalog.find((i) => i.id === id)?.name)
                                .filter(Boolean)
                                .join(' + ')}
                            </p>
                            <p className="font-display mt-4 text-[9px] font-bold uppercase tracking-[0.22em] text-purple-300/80">
                              Effectiveness
                            </p>
                            <p className="font-data mt-1 text-3xl font-bold tabular-nums tracking-tight text-amber-100 md:text-4xl [text-shadow:0_0_28px_rgba(251,191,36,0.35)]">
                              <span className="tabular-nums">{mostLethal.impactEfficiencyPct}%</span>{' '}
                              <span className="text-lg font-semibold text-purple-200/95 md:text-xl">
                                faster than the average time for this boss
                              </span>
                            </p>
                            {baselineMostLethal ? (
                              <DeltaIndicator
                                kind="percent"
                                baseline={baselineMostLethal.impactEfficiencyPct}
                                current={mostLethal.impactEfficiencyPct}
                              />
                            ) : null}
                            <p className="font-data mt-2 text-[10px] text-slate-500">
                              Average time with this item combination:{' '}
                              <span className="tabular-nums text-slate-300">
                                {formatSecondsAsHMS(mostLethal.meanSecForCombo)}
                              </span>
                              {mostLethal.source === 'historical' ? (
                                <span className="text-slate-600">
                                  {' '}
                                  · <span className="tabular-nums">{mostLethal.comboSampleCount}</span> recorded boss
                                  fight{mostLethal.comboSampleCount === 1 ? '' : 's'}
                                </span>
                              ) : null}
                            </p>
                            <p className="font-data mt-3 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                              Based on aggregate analysis of{' '}
                              <span className="tabular-nums text-slate-400">{mostLethal.totalEncounters}</span>{' '}
                              encounters.
                            </p>
                          </>
                        )}
                      </div>
                    </motion.section>

                    <section className="rounded-2xl border border-slate-800/90 bg-slate-950/40 p-4 ring-1 ring-cyan-500/15 backdrop-blur-md md:p-6">
                      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <Zap className="h-5 w-5 text-cyan-400" strokeWidth={1.25} aria-hidden />
                          <div>
                            <h3 className="font-display text-sm font-bold uppercase tracking-[0.16em] text-cyan-100 md:text-base">
                              Test item combinations
                            </h3>
                            <p className="font-data mt-1 text-[10px] text-slate-500">
                              Select up to 5 items to test
                            </p>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={clearSimLoadout}
                          className="flex items-center gap-2 rounded-lg border border-slate-600 bg-slate-900/80 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-300 transition hover:border-cyan-500/40 hover:text-cyan-200"
                        >
                          <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
                          Clear all items
                        </button>
                      </div>

                      <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:gap-8">
                        <div className="min-w-0 flex-1 space-y-4">
                        <div>
                          <p className="font-display mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                            Loadout
                          </p>
                          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                            {simSlots.map((slotId, slotIndex) => {
                              const item = slotId != null ? itemsCatalog.find((i) => i.id === slotId) : null;
                              return (
                                <motion.button
                                  key={`slot-${slotIndex}`}
                                  type="button"
                                  animate={
                                    item
                                      ? {
                                          boxShadow: [
                                            '0 0 0 0 rgba(34,211,238,0)',
                                            '0 0 28px rgba(34,211,238,0.35)',
                                            '0 0 0 0 rgba(34,211,238,0)',
                                          ],
                                        }
                                      : {}
                                  }
                                  transition={
                                    item ? { duration: 2.2, repeat: Infinity, ease: 'easeInOut' } : undefined
                                  }
                                  onClick={() => {
                                    if (slotId != null) {
                                      setSimSlots((s) => {
                                        const next = [...s];
                                        next[slotIndex] = null;
                                        return next;
                                      });
                                    }
                                  }}
                                  className={`relative flex h-24 w-full min-w-0 flex-col items-center justify-center rounded-xl border-2 border-dashed transition sm:h-28 ${
                                    item
                                      ? 'border-cyan-400/55 bg-slate-900/90 shadow-[0_0_24px_rgba(34,211,238,0.2),inset_0_0_24px_rgba(34,211,238,0.06)]'
                                      : 'border-slate-600 bg-slate-950/80 hover:border-slate-500'
                                  } `}
                                  aria-label={
                                    item ? `Remove ${item.name} from slot ${slotIndex + 1}` : `Empty slot ${slotIndex + 1}`
                                  }
                                >
                                  {item ? (
                                    <>
                                      <span
                                        className="h-3 w-3 rounded-full"
                                        style={itemAccentDotStyle(item.id, true)}
                                        aria-hidden
                                      />
                                      <span className="font-display mt-2 max-w-22 truncate px-1 text-center text-[9px] font-bold uppercase tracking-wide text-slate-200">
                                        {item.name}
                                      </span>
                                    </>
                                  ) : (
                                    <span className="font-data text-[10px] uppercase tracking-wider text-slate-600">
                                      Empty
                                    </span>
                                  )}
                                </motion.button>
                              );
                            })}
                          </div>
                        </div>

                        <div className="flex max-h-[min(420px,52vh)] flex-col overflow-hidden rounded-xl border border-slate-800/90 bg-slate-950/75 shadow-[inset_0_0_40px_rgba(0,0,0,0.35)] backdrop-blur-md">
                          <div className="shrink-0 border-b border-slate-800/80 p-3">
                            <label className="relative block">
                              <Search
                                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-500/60"
                                strokeWidth={1.5}
                                aria-hidden
                              />
                              <input
                                type="search"
                                value={libraryQuery}
                                onChange={(e) => setLibraryQuery(e.target.value)}
                                placeholder="Search item list…"
                                className="w-full rounded-lg border border-cyan-500/35 bg-slate-950/85 py-2.5 pl-10 pr-3 font-data text-sm text-slate-100 placeholder:text-slate-600 shadow-[0_0_20px_rgba(34,211,238,0.06)] outline-none ring-0 transition focus:border-cyan-400/65 focus:shadow-[0_0_24px_rgba(34,211,238,0.12)]"
                              />
                            </label>
                          </div>
                          <div className="min-h-0 flex-1 overflow-y-auto p-3 [scrollbar-color:rgba(51,65,85,0.85)_transparent]">
                            <p className="font-display mb-2 px-0.5 text-[8px] font-bold uppercase tracking-[0.2em] text-slate-600">
                              Item list
                            </p>
                            {filteredLibraryItems.length === 0 ? (
                              <p className="font-data py-8 text-center text-sm text-slate-500">No items match.</p>
                            ) : (
                              <ul className="space-y-1 pb-4">
                                {filteredLibraryItems.map((item) => {
                                  const inLoadout = simSlots.includes(item.id);
                                  return (
                                    <li key={item.id}>
                                      <button
                                        type="button"
                                        onClick={() => toggleSimItem(item.id)}
                                        className={`flex w-full items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition ${
                                          inLoadout
                                            ? 'border-cyan-400/55 bg-cyan-500/15 shadow-[0_0_14px_rgba(34,211,238,0.15)]'
                                            : 'border-slate-700/90 bg-slate-900/60 hover:border-slate-500 hover:bg-slate-900/85'
                                        }`}
                                        aria-label={`${inLoadout ? 'Remove' : 'Equip'} ${item.name}`}
                                      >
                                        <span
                                          className="h-2 w-2 shrink-0 rounded-full"
                                          style={itemAccentDotStyle(item.id, inLoadout)}
                                          aria-hidden
                                        />
                                        <span className="font-display min-w-0 flex-1 truncate text-[11px] font-bold uppercase tracking-wide text-slate-200">
                                          {item.name}
                                        </span>
                                        {item.popularity != null ? (
                                          <span className="font-data shrink-0 text-[10px] tabular-nums text-slate-500">
                                            {item.popularity}%
                                          </span>
                                        ) : null}
                                      </button>
                                    </li>
                                  );
                                })}
                              </ul>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="w-full shrink-0 xl:sticky xl:top-3 xl:w-[min(100%,22rem)]">
                        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 backdrop-blur-sm md:p-5">
                          <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400">
                            Expected result
                          </h4>
                          <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
                            <div>
                              <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">
                                Average fight time
                              </p>
                              <p className="font-data mt-1 text-xl font-bold tabular-nums text-slate-200">
                                {globalAvgSec > 0 ? globalAvgLabel : '—'}
                              </p>
                            </div>
                            <div>
                              <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">
                                Time with these items
                              </p>
                              <p
                                className={`font-data mt-1 text-xl font-bold tabular-nums ${
                                  equippedIds.length === 0
                                    ? 'text-slate-400'
                                    : synergyFaster
                                      ? 'text-cyan-300'
                                      : 'text-orange-300'
                                }`}
                              >
                                {globalAvgSec > 0 ? formatSecondsAsHMS(synergyProjectedSec) : '—'}
                              </p>
                              {equippedIds.length > 0 && globalAvgSec > 0 && itemImpactEstimate.hasEstimate && (
                                <p className="font-data mt-2 text-[10px] tabular-nums text-slate-500">
                                  Data delta:{' '}
                                  <span className={synergyFaster ? 'text-cyan-400' : 'text-orange-400'}>
                                    {synergyDeltaPct > 0 ? '−' : synergyDeltaPct < 0 ? '+' : ''}
                                    {Math.abs(synergyDeltaPct)}% vs baseline
                                  </span>
                                </p>
                              )}
                            </div>
                          </div>

                          <p className="font-display mt-4 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-600">
                            Performance comparison
                          </p>
                          <div className="mt-2 h-[200px] w-full min-w-0 sm:h-[220px]">
                            {globalAvgSec <= 0 ? (
                              <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-800">
                                <p className="font-data px-2 text-center text-sm text-slate-500">
                                  No baseline duration for comparison.
                                </p>
                              </div>
                            ) : (
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={compareBarData} margin={{ top: 8, right: 8, left: 4, bottom: 48 }}>
                                  <defs>
                                    <linearGradient id={simCompareChartId} x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.85} />
                                      <stop offset="100%" stopColor="#0891b2" stopOpacity={0.5} />
                                    </linearGradient>
                                  </defs>
                                  <CartesianGrid
                                    strokeDasharray="3 6"
                                    stroke="#334155"
                                    strokeOpacity={0.5}
                                    vertical={false}
                                  />
                                  <XAxis
                                    dataKey="label"
                                    tick={{ fill: '#94a3b8', fontSize: 8, fontFamily: 'JetBrains Mono, monospace' }}
                                    axisLine={{ stroke: '#475569' }}
                                    tickLine={{ stroke: '#475569' }}
                                    interval={0}
                                    angle={-14}
                                    textAnchor="end"
                                    height={48}
                                  />
                                  <YAxis
                                    tick={{ fill: '#94a3b8', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
                                    axisLine={{ stroke: '#475569' }}
                                    tickLine={{ stroke: '#475569' }}
                                    label={{
                                      value: 'Minutes',
                                      angle: -90,
                                      position: 'insideLeft',
                                      fill: '#64748b',
                                      fontSize: 9,
                                      fontFamily: 'JetBrains Mono, monospace',
                                    }}
                                  />
                                  <Tooltip
                                    cursor={{ fill: 'rgba(51, 65, 85, 0.2)' }}
                                    content={({ active, payload }) => {
                                      if (!active || !payload?.length) return null;
                                      const row = payload[0].payload;
                                      return (
                                        <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                                          <p className="font-data text-[10px] font-semibold text-slate-300">{row.label}</p>
                                          <p className="font-data mt-1 tabular-nums text-sm text-white">
                                            {formatSecondsAsHMS(row.seconds)}
                                          </p>
                                          <p className="font-data text-[10px] text-slate-500">{row.minutes} min</p>
                                        </div>
                                      );
                                    }}
                                  />
                                  <Bar dataKey="minutes" radius={[6, 6, 0, 0]} maxBarSize={48}>
                                    {compareBarData.map((row) => (
                                      <Cell
                                        key={row.key}
                                        fill={
                                          row.key === 'synergy' && equippedIds.length > 0 && synergyFaster
                                            ? `url(#${simCompareChartId})`
                                            : row.fill
                                        }
                                      />
                                    ))}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            )}
                          </div>
                          <p className="font-data mt-2 text-[9px] leading-relaxed text-slate-600">
                            Cyan bar = faster than the average time for this boss; orange = slower. Updates automatically as you change
                            items.
                          </p>
                          {equippedIds.length > 0 && !itemImpactEstimate.hasEstimate ? (
                            <p className="font-data mt-2 text-[10px] text-slate-500">
                              Insufficient historical data to estimate item impact.
                            </p>
                          ) : null}
                        </div>
                      </div>
                    </div>
                    </section>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </div>
    </div>
  );
}
