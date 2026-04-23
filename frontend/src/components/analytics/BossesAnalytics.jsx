import { useEffect, useId, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { Award, ChartColumn, Crosshair, Globe, Radar, Swords, Target, Timer } from 'lucide-react';
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

/**
 * BOSSES — Master-detail tactical intel. Global aggregates from dashboard.bosses + runsHistory.
 */
export default function BossesAnalytics({ data }) {
  const dashboard = data.dashboard;
  const bosses = dashboard.bosses ?? [];
  const [selectedBossId, setSelectedBossId] = useState(null);
  const chartGradId = useId().replace(/:/g, '');

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
          <code className="text-cyan-600/90">dashboard.bosses</code> +{' '}
          <code className="text-cyan-600/90">dashboard.runsHistory</code>
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
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </div>
    </div>
  );
}
