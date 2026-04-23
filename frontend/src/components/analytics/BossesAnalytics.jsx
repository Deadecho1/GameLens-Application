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
  ReferenceLine,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Award,
  Bomb,
  ChartColumn,
  Crosshair,
  FlaskConical,
  Gem,
  Globe,
  Hammer,
  Heart,
  Layers2,
  LayoutGrid,
  Package,
  Radar,
  RotateCcw,
  Search,
  Shield,
  Sparkles,
  Sword,
  Swords,
  Target,
  Timer,
  Wand2,
  Wrench,
  Zap,
} from 'lucide-react';
import { durationToSeconds, formatSecondsAsHMS, secondsToMinutes } from '../../utils/duration';

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

/** Longer mean lifespan → better rank (rank 1 = longest). Top % = ceil(rank / N * 100). */
function survivalRankTopPercent(bossId, dashboard) {
  const bosses = dashboard.bosses ?? [];
  if (!bosses.length) return { topPct: null, rank: null, total: 0, singleCohort: false };
  if (bosses.length === 1) {
    return { topPct: null, rank: 1, total: 1, singleCohort: true };
  }

  const scored = bosses.map((b) => ({
    id: b.id,
    mean: meanSeconds(collectGlobalLifespanSeconds(b.id, dashboard)),
  }));
  const sorted = [...scored].sort((a, b) => b.mean - a.mean);
  const rank = sorted.findIndex((x) => x.id === bossId) + 1;
  const topPct = Math.max(1, Math.ceil((rank / bosses.length) * 100));
  return { topPct, rank, total: bosses.length, singleCohort: false };
}

function pickItemIcon(name) {
  const n = String(name ?? '').toLowerCase();
  if (n.includes('mallet') || n.includes('hammer')) return Hammer;
  if (n.includes('dagger') || n.includes('sword')) return Sword;
  if (n.includes('seed') || n.includes('explosive')) return Bomb;
  if (n.includes('charm') || n.includes('void')) return Gem;
  if (n.includes('healing') || n.includes('draught')) return Heart;
  if (n.includes('focus') || n.includes('arcane')) return Wand2;
  if (n.includes('venom') || n.includes('flask') || n.includes('potion')) return FlaskConical;
  if (n.includes('buckler') || n.includes('shield') || n.includes('plate')) return Shield;
  if (n.includes('power')) return Zap;
  return Package;
}

function inferItemCategory(item) {
  if (item.category) return item.category;
  const n = String(item.name ?? '').toLowerCase();
  if (/(shield|plate|buckler)/.test(n)) return 'defensive';
  if (/(sword|dagger|mallet|venom|explosive|thunder|fire|frost)/.test(n)) return 'offensive';
  return 'utility';
}

function itemLogicTag(item) {
  if (item.logicTag) return item.logicTag;
  const imp = String(item.impact ?? '').toLowerCase();
  if (imp === 'high') return 'High impact';
  if (imp === 'low') return 'Support chip';
  return 'Balanced kit';
}

/** Client-side mock: positive total = shorter projected engagement (percent points vs baseline). */
function mockSynergyTimeReductionPct(itemIds, catalog) {
  let pct = 0;
  for (const id of itemIds) {
    if (id == null) continue;
    const item = catalog.find((i) => i.id === id);
    if (!item) continue;
    const cat = inferItemCategory(item);
    if (cat === 'offensive') pct += 15;
    else if (cat === 'defensive') pct -= 5;
    else if (cat === 'utility') pct += 8;
    else pct += 5;
  }
  return pct;
}

const LIBRARY_CATEGORY_TABS = [
  { id: 'all', label: 'All', Icon: LayoutGrid },
  { id: 'offensive', label: 'Offensive', Icon: Swords },
  { id: 'defensive', label: 'Defensive', Icon: Shield },
  { id: 'utility', label: 'Utility', Icon: Wrench },
];

/** Cross-reference dashboard.items with boss gearSynergies + itemEffectiveness. */
function buildGearAnalysis(selectedBoss, items) {
  const catalog = items ?? [];
  const byId = Object.fromEntries(catalog.map((i) => [i.id, i]));
  if (!selectedBoss) return { synergies: [], itemRows: [], effDomain: [0, 1] };

  const synergies = (selectedBoss.gearSynergies ?? []).map((s) => ({
    label: (s.itemIds ?? []).map((id) => byId[id]?.name ?? `#${id}`).join(' + '),
    timeReductionPct: Number(s.timeReductionPct) || 0,
  }));

  const itemRows = (selectedBoss.itemEffectiveness ?? [])
    .map((row) => {
      const item = byId[row.itemId];
      const eff = Number(row.timeReductionVsGlobalPct) || 0;
      let fill = '#64748b';
      if (eff >= 6) fill = '#34d399';
      else if (eff <= -2) fill = '#f87171';
      return {
        id: row.itemId,
        name: item?.name ?? `Item ${row.itemId}`,
        effectiveness: eff,
        fill,
        correlationLabel:
          eff >= 6 ? 'High Threat to Boss' : eff <= -2 ? 'Extended engagement' : 'Neutral footprint',
      };
    })
    .sort((a, b) => b.effectiveness - a.effectiveness);

  const vals = itemRows.map((r) => r.effectiveness);
  let effDomain = [0, 1];
  if (vals.length) {
    const min = Math.min(0, ...vals);
    const max = Math.max(0, ...vals);
    const span = max - min || 10;
    const pad = Math.max(2, span * 0.12);
    effDomain = [min - pad, max + pad];
  }

  return { synergies, itemRows, effDomain };
}

/**
 * BOSSES — Master-detail tactical intel. dashboard.bosses + runsHistory + dashboard.items (gear).
 */
export default function BossesAnalytics({ data }) {
  const dashboard = data.dashboard;
  const bosses = dashboard.bosses ?? [];
  const itemsCatalog = dashboard.items ?? [];
  const [selectedBossId, setSelectedBossId] = useState(null);
  /** Synergy Lab: up to 3 equipped item ids (inventory slots). */
  const [simSlots, setSimSlots] = useState([null, null, null]);
  const [libraryQuery, setLibraryQuery] = useState('');
  const [libraryCategory, setLibraryCategory] = useState('all');
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
    setSimSlots([null, null, null]);
    setLibraryQuery('');
    setLibraryCategory('all');
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
        ? survivalRankTopPercent(selected.id, dashboard)
        : { topPct: null, rank: null, total: 0, singleCohort: false },
    [selected, dashboard]
  );

  const gearAnalysis = useMemo(
    () => buildGearAnalysis(selected, itemsCatalog),
    [selected, itemsCatalog]
  );

  const impactChartHeight = Math.max(200, gearAnalysis.itemRows.length * 44);

  const equippedIds = useMemo(() => simSlots.filter((id) => id != null), [simSlots]);
  const synergyReductionPct = useMemo(
    () => mockSynergyTimeReductionPct(equippedIds, itemsCatalog),
    [equippedIds, itemsCatalog]
  );
  const synergyProjectedSec = useMemo(() => {
    if (globalAvgSec <= 0) return 0;
    if (equippedIds.length === 0) return globalAvgSec;
    const raw = Math.round(globalAvgSec * (1 - synergyReductionPct / 100));
    return Math.max(15, raw);
  }, [globalAvgSec, equippedIds.length, synergyReductionPct]);
  const synergyFaster = synergyProjectedSec < globalAvgSec;
  const compareBarData = useMemo(() => {
    const baseFill = '#575f6b';
    const synergyFill = equippedIds.length === 0 ? '#64748b' : synergyFaster ? '#22d3ee' : '#fb923c';
    return [
      {
        key: 'base',
        label: 'Global average (base)',
        seconds: globalAvgSec,
        minutes: secondsToMinutes(globalAvgSec),
        fill: baseFill,
      },
      {
        key: 'synergy',
        label: 'Synergy projection',
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

  const clearSimLoadout = () => setSimSlots([null, null, null]);

  const filteredLibraryItems = useMemo(() => {
    const q = libraryQuery.trim().toLowerCase();
    return itemsCatalog.filter((item) => {
      const cat = inferItemCategory(item);
      if (libraryCategory !== 'all' && cat !== libraryCategory) return false;
      if (q && !String(item.name ?? '').toLowerCase().includes(q)) return false;
      return true;
    });
  }, [itemsCatalog, libraryQuery, libraryCategory]);

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <Swords className="h-5 w-5 text-cyan-400" strokeWidth={1.25} aria-hidden />
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-cyan-500/80">
            Boss intelligence
          </p>
        </div>
        <h3 className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl">
          Tactical command · master detail
        </h3>
        <p className="font-data mt-2 text-sm text-slate-500">
          Global aggregates from{' '}
          <code className="text-cyan-600/90">dashboard.bosses</code>,{' '}
          <code className="text-cyan-600/90">dashboard.runsHistory</code>, and{' '}
          <code className="text-cyan-600/90">dashboard.items</code>
        </p>
      </header>

      <div className="flex min-h-[min(640px,75vh)] flex-col gap-4 lg:flex-row lg:gap-0 lg:rounded-2xl lg:border lg:border-slate-800/90 lg:bg-slate-950/40 lg:shadow-[inset_0_1px_0_rgba(34,211,238,0.06)]">
        <aside className="flex w-full flex-col border-slate-800/80 lg:w-[25%] lg:min-w-[220px] lg:max-w-[320px] lg:border-r lg:bg-slate-950/50">
          <div className="flex items-center gap-2 border-b border-slate-800/80 px-4 py-3">
            <Radar className="h-4 w-4 text-cyan-400/90" strokeWidth={1.25} aria-hidden />
            <span className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
              Target rail
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
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border ${
                        active
                          ? 'border-cyan-400/40 bg-cyan-500/15 text-cyan-200'
                          : 'border-slate-700 bg-slate-950/60 text-slate-500 group-hover:text-slate-300'
                      }`}
                    >
                      {active ? (
                        <Target className="h-4 w-4" strokeWidth={1.5} aria-hidden />
                      ) : (
                        <Crosshair className="h-4 w-4 opacity-70" strokeWidth={1.25} aria-hidden />
                      )}
                    </span>
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
                  <p className="font-data text-sm text-slate-500">Select a boss from the rail.</p>
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
                    {(survivalRank.topPct != null || survivalRank.singleCohort) && (
                      <div className="flex items-center gap-3 rounded-xl border border-slate-700/90 bg-slate-950/55 px-4 py-3 ring-1 ring-cyan-500/10 backdrop-blur-md">
                        <Award className="h-8 w-8 shrink-0 text-cyan-400/90" strokeWidth={1.15} aria-hidden />
                        <div>
                          <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                            Survival rank
                          </p>
                          {survivalRank.singleCohort ? (
                            <p className="font-data text-sm text-slate-400">Baseline — single boss in dataset</p>
                          ) : (
                            <>
                              <p className="font-data text-lg font-bold tabular-nums text-cyan-100">
                                Top {survivalRank.topPct}%
                              </p>
                              <p className="font-data text-[10px] tabular-nums text-slate-500">
                                #{survivalRank.rank} of {survivalRank.total} by mean lifespan
                              </p>
                            </>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
                    <div className="rounded-2xl border border-cyan-500/20 bg-slate-900/50 p-6 shadow-[0_0_40px_rgba(34,211,238,0.08),inset_0_1px_0_rgba(255,255,255,0.04)] ring-1 ring-cyan-500/10 backdrop-blur-md">
                      <div className="mb-4 flex items-center gap-2">
                        <Timer className="h-5 w-5 text-cyan-300" strokeWidth={1.25} aria-hidden />
                        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-200/90">
                          Global average lifespan
                        </h3>
                      </div>
                      <p className="font-data text-[10px] uppercase tracking-wider text-slate-500">
                        Mean encounter duration · full dataset
                      </p>
                      <p
                        className="mt-4 font-data text-4xl font-bold tabular-nums tracking-tight text-cyan-100 md:text-5xl lg:text-6xl"
                        style={{ textShadow: '0 0 32px rgba(34, 211, 238, 0.35)' }}
                      >
                        {globalAvgLabel}
                      </p>
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
                      <p className="font-data text-[10px] uppercase tracking-wider text-slate-600">
                        Encounter count
                      </p>
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 rounded-2xl border border-slate-800/90 bg-slate-950/35 p-4 backdrop-blur-md md:p-5">
                    <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <ChartColumn className="h-4 w-4 text-cyan-400/90" strokeWidth={1.25} aria-hidden />
                        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-white/90">
                          Lifespan frequency distribution
                        </h3>
                      </div>
                      <span className="font-data text-[10px] text-slate-500">Time bracket · occurrences</span>
                    </div>
                    <p className="font-data mb-4 text-[10px] text-slate-600">
                      Encounters bucketed by fight length — highlights consistency vs. variance in the global
                      dataset.
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
                                value: 'Occurrences',
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
                                      {row.count} occurrence{row.count === 1 ? '' : 's'}
                                    </p>
                                  </div>
                                );
                              }}
                            />
                            <Bar
                              dataKey="count"
                              name="Occurrences"
                              fill={`url(#${chartGradId})`}
                              radius={[6, 6, 0, 0]}
                              maxBarSize={72}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </div>

                  <section className="rounded-2xl border border-slate-800/90 bg-slate-950/40 p-4 ring-1 ring-cyan-500/10 backdrop-blur-md md:p-6">
                    <div className="mb-1 flex flex-wrap items-center gap-2">
                      <Layers2 className="h-4 w-4 text-cyan-400" strokeWidth={1.25} aria-hidden />
                      <p className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-slate-500">
                        {'Gear & synergy impact'}
                      </p>
                    </div>
                    <h3 className="font-display text-sm font-bold uppercase tracking-[0.18em] text-cyan-100/95 md:text-base">
                      Equipment influence on lifespan
                    </h3>
                    <p className="font-data mt-2 max-w-3xl text-[10px] leading-relaxed text-slate-500">
                      Correlation vs global average encounter length for this target. Positive time reduction
                      indicates shorter engagements when the loadout is present.
                    </p>

                    <div className="mt-6 grid gap-8 lg:grid-cols-2 lg:gap-10">
                      <div>
                        <div className="mb-3 flex items-center gap-2">
                          <Sparkles className="h-4 w-4 text-cyan-500/80" strokeWidth={1.25} aria-hidden />
                          <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                            Synergy table
                          </h4>
                        </div>
                        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950/50">
                          {gearAnalysis.synergies.length === 0 ? (
                            <p className="font-data p-4 text-sm text-slate-500">No combination data linked.</p>
                          ) : (
                            <table className="w-full text-left font-data text-xs">
                              <thead>
                                <tr className="border-b border-slate-800 bg-slate-900/80 text-[10px] uppercase tracking-wider text-slate-500">
                                  <th className="px-3 py-2.5 font-medium">Item combination</th>
                                  <th className="px-3 py-2.5 font-medium tabular-nums">Time reduction %</th>
                                </tr>
                              </thead>
                              <tbody>
                                {gearAnalysis.synergies.map((row) => (
                                  <tr
                                    key={row.label}
                                    className="border-b border-slate-800/80 last:border-0 hover:bg-slate-900/40"
                                  >
                                    <td className="px-3 py-3">
                                      <span className="flex items-center gap-2 text-slate-200">
                                        <Sparkles className="h-3.5 w-3.5 shrink-0 text-cyan-500/70" aria-hidden />
                                        {row.label}
                                      </span>
                                    </td>
                                    <td className="px-3 py-3 tabular-nums text-cyan-200">{row.timeReductionPct}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          )}
                        </div>
                      </div>

                      <div>
                        <div className="mb-3 flex items-center gap-2">
                          <ChartColumn className="h-4 w-4 text-emerald-500/80" strokeWidth={1.25} aria-hidden />
                          <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                            Item effectiveness
                          </h4>
                        </div>
                        <p className="font-data mb-2 text-[10px] text-slate-600">
                          Horizontal axis: time reduction % vs global average (green shortens, red prolongs).
                        </p>
                        <div style={{ height: impactChartHeight }} className="w-full min-w-0">
                          {gearAnalysis.itemRows.length === 0 ? (
                            <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-800">
                              <p className="font-data text-sm text-slate-500">No per-item effectiveness rows.</p>
                            </div>
                          ) : (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                layout="vertical"
                                data={gearAnalysis.itemRows}
                                margin={{ top: 4, right: 12, left: 4, bottom: 4 }}
                              >
                                <CartesianGrid
                                  strokeDasharray="3 6"
                                  stroke="#334155"
                                  strokeOpacity={0.45}
                                  horizontal={false}
                                />
                                <XAxis
                                  type="number"
                                  domain={gearAnalysis.effDomain}
                                  tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                                  axisLine={{ stroke: '#475569' }}
                                  tickLine={{ stroke: '#475569' }}
                                  tickFormatter={(v) => `${v}%`}
                                  label={{
                                    value: 'Time reduction %',
                                    position: 'insideBottom',
                                    offset: -2,
                                    fill: '#64748b',
                                    fontSize: 10,
                                    fontFamily: 'JetBrains Mono, monospace',
                                  }}
                                />
                                <YAxis
                                  type="category"
                                  dataKey="name"
                                  width={108}
                                  tick={{ fill: '#e2e8f0', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                                  axisLine={{ stroke: '#475569' }}
                                  tickLine={false}
                                />
                                <Tooltip
                                  cursor={{ fill: 'rgba(51, 65, 85, 0.15)' }}
                                  content={({ active, payload }) => {
                                    if (!active || !payload?.length) return null;
                                    const row = payload[0].payload;
                                    const TipIcon = pickItemIcon(row.name);
                                    return (
                                      <div className="max-w-xs rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                                        <div className="flex items-center gap-2">
                                          <TipIcon className="h-4 w-4 text-cyan-400" aria-hidden />
                                          <p className="font-data text-xs font-semibold text-white">{row.name}</p>
                                        </div>
                                        <p className="font-data mt-1 tabular-nums text-sm text-cyan-200">
                                          {row.effectiveness > 0 ? '+' : ''}
                                          {row.effectiveness}% vs average
                                        </p>
                                        <p className="font-data mt-1 text-[10px] text-slate-500">{row.correlationLabel}</p>
                                      </div>
                                    );
                                  }}
                                />
                                <ReferenceLine
                                  x={0}
                                  stroke="#475569"
                                  strokeDasharray="4 4"
                                  strokeOpacity={0.9}
                                />
                                <Bar dataKey="effectiveness" name="Effectiveness" radius={[0, 4, 4, 0]} barSize={18}>
                                  {gearAnalysis.itemRows.map((row) => (
                                    <Cell key={row.id} fill={row.fill} stroke={row.fill} strokeOpacity={0.35} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          )}
                        </div>
                        <div className="mt-3 flex flex-wrap gap-4 font-data text-[10px] text-slate-500">
                          <span className="flex items-center gap-1.5">
                            <span className="h-2 w-4 rounded-sm bg-emerald-400/90" /> Shortens engagement
                          </span>
                          <span className="flex items-center gap-1.5">
                            <span className="h-2 w-4 rounded-sm bg-slate-500" /> Neutral
                          </span>
                          <span className="flex items-center gap-1.5">
                            <span className="h-2 w-4 rounded-sm bg-red-400/90" /> Prolongs engagement
                          </span>
                        </div>
                      </div>
                    </div>
                  </section>

                  <section className="rounded-2xl border border-slate-800/90 bg-slate-950/40 p-4 ring-1 ring-cyan-500/15 backdrop-blur-md md:p-6">
                    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Zap className="h-5 w-5 text-cyan-400" strokeWidth={1.25} aria-hidden />
                        <div>
                          <h3 className="font-display text-sm font-bold uppercase tracking-[0.16em] text-cyan-100 md:text-base">
                            {'Synergy Lab: Item Impact Simulator'}
                          </h3>
                          <p className="font-data mt-1 text-[10px] text-slate-500">
                            Scalable armory · mock projection · max 3 equipped
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={clearSimLoadout}
                        className="flex items-center gap-2 rounded-lg border border-slate-600 bg-slate-900/80 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-300 transition hover:border-cyan-500/40 hover:text-cyan-200"
                      >
                        <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
                        Clear all
                      </button>
                    </div>

                    <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:gap-8">
                      <div className="min-w-0 flex-1 space-y-4">
                        <div>
                          <p className="font-display mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                            Loadout
                          </p>
                          <div className="flex flex-wrap justify-center gap-3 sm:justify-start">
                            {simSlots.map((slotId, slotIndex) => {
                              const item = slotId != null ? itemsCatalog.find((i) => i.id === slotId) : null;
                              const Icon = item ? pickItemIcon(item.name) : Package;
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
                                  className={`relative flex h-24 w-24 flex-col items-center justify-center rounded-xl border-2 border-dashed transition sm:h-28 sm:w-28 ${
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
                                      <Icon className="h-8 w-8 text-cyan-200" strokeWidth={1.15} aria-hidden />
                                      <span className="font-data mt-2 max-w-22 truncate px-1 text-center text-[9px] text-slate-400">
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
                                placeholder="Search item library…"
                                className="w-full rounded-lg border border-cyan-500/35 bg-slate-950/85 py-2.5 pl-10 pr-3 font-data text-sm text-slate-100 placeholder:text-slate-600 shadow-[0_0_20px_rgba(34,211,238,0.06)] outline-none ring-0 transition focus:border-cyan-400/65 focus:shadow-[0_0_24px_rgba(34,211,238,0.12)]"
                              />
                            </label>
                            <div className="mt-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Item categories">
                              {LIBRARY_CATEGORY_TABS.map((tab) => {
                                const active = libraryCategory === tab.id;
                                const TabIcon = tab.Icon;
                                return (
                                  <button
                                    key={tab.id}
                                    type="button"
                                    role="tab"
                                    aria-selected={active}
                                    onClick={() => setLibraryCategory(tab.id)}
                                    className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-display text-[8px] font-bold uppercase tracking-[0.15em] transition ${
                                      active
                                        ? 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200 shadow-[0_0_14px_rgba(34,211,238,0.2)]'
                                        : 'border-slate-700/90 bg-slate-900/50 text-slate-500 hover:border-slate-600 hover:text-slate-300'
                                    } `}
                                  >
                                    <TabIcon className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
                                    {tab.label}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                          <div className="min-h-0 flex-1 overflow-y-auto p-3 [scrollbar-color:rgba(51,65,85,0.85)_transparent]">
                            <p className="font-display mb-2 px-0.5 text-[8px] font-bold uppercase tracking-[0.2em] text-slate-600">
                              Item library
                            </p>
                            {filteredLibraryItems.length === 0 ? (
                              <p className="font-data py-8 text-center text-sm text-slate-500">No items match filters.</p>
                            ) : (
                              <div className="grid grid-cols-4 gap-2 pb-16 sm:grid-cols-5 md:grid-cols-6">
                                {filteredLibraryItems.map((item) => {
                                  const inLoadout = simSlots.includes(item.id);
                                  const Icon = pickItemIcon(item.name);
                                  return (
                                    <div key={item.id} className="group relative aspect-square">
                                      <button
                                        type="button"
                                        onClick={() => toggleSimItem(item.id)}
                                        className={`relative flex h-full w-full flex-col items-center justify-center rounded-lg border p-1 transition ${
                                          inLoadout
                                            ? 'border-cyan-400/55 bg-cyan-500/15 shadow-[0_0_18px_rgba(34,211,238,0.25)]'
                                            : 'border-slate-700/90 bg-slate-900/60 hover:border-slate-500 hover:bg-slate-900/85'
                                        } `}
                                        aria-label={`${inLoadout ? 'Remove' : 'Equip'} ${item.name}`}
                                      >
                                        <Icon className="h-6 w-6 text-cyan-300/95 sm:h-7 sm:w-7" strokeWidth={1.1} aria-hidden />
                                      </button>
                                      <div className="pointer-events-none absolute left-1/2 top-full z-80 mt-1 w-max max-w-52 -translate-x-1/2 scale-95 rounded-lg border border-cyan-500/25 bg-slate-950/95 px-2.5 py-2 opacity-0 shadow-[0_12px_40px_rgba(0,0,0,0.55)] backdrop-blur-md transition duration-150 group-hover:scale-100 group-hover:opacity-100">
                                        <p className="font-display text-[10px] font-bold uppercase tracking-wide text-cyan-100">
                                          {item.name}
                                        </p>
                                        <p className="font-data mt-1 text-[10px] tabular-nums text-slate-400">
                                          Popularity <span className="text-slate-200">{item.popularity}</span>
                                        </p>
                                        <p className="font-data mt-1 border-t border-slate-800 pt-1 text-[9px] text-slate-500">
                                          <span className="text-cyan-500/90">Logic</span>{' '}
                                          <span className="text-slate-300">{itemLogicTag(item)}</span>
                                        </p>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="w-full shrink-0 xl:sticky xl:top-3 xl:w-[min(100%,22rem)]">
                        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 backdrop-blur-sm md:p-5">
                          <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400">
                            Predicted combat outcome
                          </h4>
                          <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
                            <div>
                              <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">
                                Global average
                              </p>
                              <p className="font-data mt-1 text-xl font-bold tabular-nums text-slate-200">
                                {globalAvgSec > 0 ? globalAvgLabel : '—'}
                              </p>
                            </div>
                            <div>
                              <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">
                                Synergy lifespan
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
                              {equippedIds.length > 0 && globalAvgSec > 0 && (
                                <p className="font-data mt-2 text-[10px] tabular-nums text-slate-500">
                                  Mock loadout delta:{' '}
                                  <span className={synergyFaster ? 'text-cyan-400' : 'text-orange-400'}>
                                    {synergyReductionPct > 0 ? '−' : synergyReductionPct < 0 ? '+' : ''}
                                    {Math.abs(synergyReductionPct)}% vs baseline
                                  </span>
                                </p>
                              )}
                            </div>
                          </div>

                          <p className="font-display mt-4 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-600">
                            Impact chart
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
                            Cyan synergy bar = faster clear vs global average; orange = slower. Updates live as you
                            change loadout.
                          </p>
                        </div>
                      </div>
                    </div>
                  </section>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </div>
    </div>
  );
}
