import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Calendar,
  Clock,
  Crosshair,
  Package,
  Swords,
  Timer,
} from 'lucide-react';
import { durationToSeconds, formatSecondsAsHMS } from '../../../utils/duration';
import {
  buildRunDurationTrendSeries,
  resolveRunTimestamp,
} from '../../../utils/runMovingAverage';
import { MemoizedRadarChart } from './TacticalRadarChart';
import RunItemChip from './RunItemChip';

/** Synergy bonus label from catalog popularity (dataStore) — scaled to seconds proxy. */
function itemSynergySeconds(item) {
  if (!item || typeof item.popularity !== 'number') return 1;
  return Math.max(1, Math.round(item.popularity / 12));
}

/**
 * Converts run `duration` strings from dataStore (e.g. "00:28:00") to seconds for charts.
 */
export function runDurationToSeconds(hms) {
  return durationToSeconds(hms);
}

function itemById(catalog, id) {
  return catalog.find((i) => i.id === id) ?? null;
}

function itemNameById(catalog, id) {
  return itemById(catalog, id)?.name ?? null;
}

function bossNameById(catalog, id) {
  return catalog.find((b) => b.id === id)?.name ?? null;
}

const glitchInjectVariants = {
  initial: { opacity: 0.96, x: 0, skewX: 0 },
  animate: {
    opacity: [1, 0.94, 1, 1],
    x: [0, -3, 2, 0],
    skewX: ['0deg', '-0.6deg', '0.4deg', '0deg'],
    transition: { duration: 0.24, times: [0, 0.15, 0.35, 1], ease: 'easeOut' },
  },
};

/**
 * Run Session Analytics — data from `dataStore.js` (`initialData` + live `data`).
 * @param {boolean} [embedded] — compact layout inside PostProcessingReviewModal (no page header).
 * @param {string|null} [initialSelectedRunId] — pre-select a run when embedded.
 */
export default function RunSessionAnalytics({
  data,
  embedded = false,
  initialSelectedRunId = null,
}) {
  const runsHistory = useMemo(() => {
    const runs = data?.dashboard?.runsHistory;
    return Array.isArray(runs) ? runs : [];
  }, [data]);

  const itemsCatalog = data?.dashboard?.items ?? [];
  const bossesCatalog = data?.dashboard?.bosses ?? [];

  const globalAverageDurationSeconds = useMemo(() => {
    if (!runsHistory.length) return 0;
    const sum = runsHistory.reduce((acc, r) => acc + runDurationToSeconds(r.duration), 0);
    return sum / runsHistory.length;
  }, [runsHistory]);

  /** Stable chart rows: chronological sort, Unix timestamps, moving avg — only when raw runs change. */
  const chartBundle = useMemo(() => {
    if (!runsHistory.length) {
      return { chartData: [], n: 0, yMin: 0, yMax: 1, yPad: 0 };
    }
    const trendSeries = buildRunDurationTrendSeries(runsHistory);
    const secs = trendSeries.map((p) => p.durationSec);
    const avgSecs = trendSeries.map((p) => p.movingAverage);
    const yMin = Math.min(...secs, ...avgSecs);
    const yMax = Math.max(...secs, ...avgSecs);
    const spread = yMax - yMin || 1;
    const yPad = Math.max(30, spread * 0.06);
    const n = trendSeries.length;
    const chartData = trendSeries.map((point, index) => ({
      ...point,
      timestamp: resolveRunTimestamp(point.run) + index,
      run_id: point.run_id ?? point.runId ?? point.run?.id ?? null,
      minSec: yMin,
      maxSec: yMax,
    }));
    return { chartData, n, yMin, yMax, yPad };
  }, [runsHistory]);

  const { chartData, n, yMin, yMax, yPad } = chartBundle;

  const runsDataKey = useMemo(() => {
    if (!runsHistory.length) return '0';
    const last = runsHistory[runsHistory.length - 1];
    return `${runsHistory.length}:${runsHistory[0]?.id}:${last?.id}`;
  }, [runsHistory]);

  const [selectedRunId, setSelectedRunId] = useState(null);
  const [hoveredRunId, setHoveredRunId] = useState(null);

  const handleSelectRun = useCallback((runId) => {
    setSelectedRunId(runId);
  }, []);

  const handleScatterHover = useCallback((runId, entering) => {
    if (entering) setHoveredRunId(runId);
    else setHoveredRunId((current) => (current === runId ? null : current));
  }, []);

  useEffect(() => {
    if (initialSelectedRunId != null) {
      setSelectedRunId(initialSelectedRunId);
      return;
    }
    if (runsHistory.length === 0) setSelectedRunId(null);
  }, [runsHistory, initialSelectedRunId]);

  const selectedRun = useMemo(
    () => runsHistory.find((r) => r.id === selectedRunId) ?? null,
    [runsHistory, selectedRunId]
  );

  const maxBossLifespanSec = useMemo(() => {
    const enc = selectedRun?.bossEncounters ?? [];
    if (!enc.length) return 1;
    return Math.max(...enc.map((e) => durationToSeconds(e.lifespan)), 1);
  }, [selectedRun]);

  const handleListSelect = useCallback((runId) => {
    setSelectedRunId(runId);
  }, []);

  const avgLabel = formatSecondsAsHMS(Math.round(globalAverageDurationSeconds));

  const chartIsHero = !selectedRunId;

  const circuitBgStyle = {
    backgroundColor: '',
    backgroundImage: `

    `,
    backgroundSize: '32px 32px, 32px 32px, 8px 8px, 8px 8px',
    backgroundPosition: '0 0, 0 0, -1px -1px, -1px -1px',
  };

  const shellClass = embedded
    ? 'relative mx-auto max-w-[1800px] px-2 py-4 md:px-6 md:py-6'
    : 'relative mx-auto max-w-[1800px] px-4 py-8 md:py-10';

  const layoutClass = embedded
    ? 'relative z-1 flex min-h-[min(480px,60vh)] flex-col bg-slate-950/30 lg:rounded-xl lg:border lg:border-slate-800'
    : 'relative z-1 flex min-h-[min(640px,72vh)] flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0 lg:rounded-2xl lg:border lg:border-slate-800 lg:bg-slate-950/45 lg:shadow-[inset_0_1px_0_rgba(148,163,184,0.05)]';

  const runSidebar = !embedded ? (
    <aside className="flex w-full flex-col border-slate-800/80 lg:w-[min(100%,280px)] lg:shrink-0 lg:border-r lg:border-slate-800/90 lg:bg-slate-950/55">
      <div className="border-b border-slate-800/80 px-4 py-3">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
          Run selection
        </p>
        <p className="font-data mt-1 text-[10px] text-slate-600">
          {runsHistory.length} session{runsHistory.length === 1 ? '' : 's'} (dataStore)
        </p>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2 [scrollbar-color:rgba(71,85,105,0.45)_transparent]">
        {runsHistory.length === 0 ? (
          <p className="font-data px-2 py-8 text-center text-sm text-slate-500">No runs in history.</p>
        ) : (
          <ul className="space-y-1">
            {runsHistory.map((run) => {
              const active = run.id === selectedRunId;
              return (
                <li key={run.id}>
                  <button
                    type="button"
                    onClick={() => handleListSelect(run.id)}
                    className={`flex w-full flex-col gap-1 rounded-lg border px-3 py-2.5 text-left transition ${
                      active
                        ? 'border-cyan-500/50 bg-cyan-500/5'
                        : 'border-slate-800 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/65'
                    }`}
                  >
                    <span className="font-display text-[11px] font-bold uppercase tracking-wide text-slate-200">
                      {run.id}
                    </span>
                    <span className="font-data flex items-center gap-1.5 text-[10px] text-slate-500">
                      <Calendar className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
                      {run.date}
                    </span>
                    <span className="font-data flex items-center gap-1.5 text-[11px] tabular-nums text-cyan-300/80">
                      <Clock className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
                      {run.duration}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  ) : null;

  return (
    <motion.div
      initial={embedded ? false : { opacity: 0, y: 14 }}
      animate={embedded ? false : { opacity: 1, y: 0 }}
      exit={embedded ? false : { opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className={shellClass}
      style={embedded ? undefined : circuitBgStyle}
    >
      {!embedded ? (
        <header className="relative z-1 mb-6">
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-cyan-500/70">
            Session intel
          </p>
          <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
            Run session analytics
          </h2>
        </header>
      ) : null}

      <div className={layoutClass}>
        {runSidebar}

        <motion.div layout className="flex min-w-0 flex-1 flex-col bg-slate-950/30">
          <motion.section
            layout
            transition={{ type: 'spring', stiffness: 320, damping: 32 }}
            className={`border-b border-slate-800/80 p-4 backdrop-blur-sm md:p-6 ${
              chartIsHero
                ? 'min-h-[min(52vh,520px)] lg:min-h-[min(56vh,560px)]'
                : 'min-h-[200px] md:min-h-[240px]'
            }`}
          >
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <Activity className="h-4 w-4 text-cyan-400/80" aria-hidden />
                <div>
                  <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-300">
                    Tactical radar · density cloud + trend
                  </h3>
                  <p className="font-data mt-1 text-[10px] text-slate-600">
                    Use tactical zoom above the chart · {n} run{n === 1 ? '' : 's'} chronological
                  </p>
                </div>
              </div>
              {selectedRunId ? (
                <button
                  type="button"
                  onClick={() => setSelectedRunId(null)}
                  className="font-display text-[8px] font-bold uppercase tracking-[0.15em] text-cyan-500/80 underline-offset-2 hover:text-cyan-300 hover:underline"
                >
                  Overview
                </button>
              ) : null}
            </div>
            {chartData.length === 0 ? (
              <div className="flex h-[280px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40 font-data text-sm text-slate-500">
                No data to plot.
              </div>
            ) : (
              <MemoizedRadarChart
                data={chartData}
                chartKey={runsDataKey}
                n={n}
                yMin={yMin}
                yMax={yMax}
                yPad={yPad}
                globalAverageDurationSeconds={globalAverageDurationSeconds}
                avgLabel={avgLabel}
                selectedRunId={selectedRunId}
                hoveredRunId={hoveredRunId}
                onSelectRun={handleSelectRun}
                onHoverRun={handleScatterHover}
                className="w-full min-w-0 rounded-xl border border-slate-800 bg-slate-950/70"
                plotHeight={chartIsHero ? 480 : 400}
              />
            )}
          </motion.section>

          <motion.section
            layout
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="relative flex-1 min-h-0 overflow-x-hidden overflow-y-auto p-4 md:p-6"
          >
            <AnimatePresence mode="wait">
              {!selectedRun ? (
                <motion.div
                  key="hint"
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 12 }}
                  transition={{ duration: 0.28 }}
                  className="flex min-h-[120px] items-center justify-center rounded-xl border border-dashed border-slate-700/80 bg-slate-950/40 px-4 py-8"
                >
                  <p className="font-data text-center text-sm text-slate-500">
                    Select a run or a shard on the radar to inject{' '}
                    <span className="text-cyan-500/80">telemetry</span>.
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key={selectedRun.id}
                  variants={glitchInjectVariants}
                  initial="initial"
                  animate="animate"
                  exit={{ opacity: 0 }}
                  className="relative rounded-2xl border border-slate-800 bg-slate-950/50 p-5"
                >
                  <div className="mb-6 border-b border-slate-800/90 pb-4">
                    <p className="font-display text-xs font-bold uppercase tracking-widest text-slate-300">
                      {selectedRun.id}
                    </p>
                    <p className="font-data mt-2 text-sm text-slate-300">
                      <span className="text-slate-500">Date:</span>{' '}
                      <span className="tabular-nums text-slate-200">{selectedRun.date}</span>
                    </p>
                    <p className="font-data mt-1 text-sm text-slate-300">
                      <span className="text-slate-500">Duration:</span>{' '}
                      <span className="tabular-nums text-cyan-300/90">{selectedRun.duration}</span>
                      <span className="ml-2 text-xs text-slate-600">
                        ({runDurationToSeconds(selectedRun.duration)}s)
                      </span>
                    </p>
                  </div>

                  <div className="flex items-center gap-2 pb-4">
                    <Timer className="h-4 w-4 text-slate-400" aria-hidden />
                    <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                      Tactical Run Trace
                    </h3>
                  </div>

                  {(selectedRun.bossEncounters ?? []).length === 0 ? (
                    <p className="font-data text-sm text-slate-600">No boss encounters for this run.</p>
                  ) : (
                    <div className="relative flex gap-4 md:gap-6">
                      <div className="relative w-8 shrink-0 md:w-10" aria-hidden>
                        <svg
                          className="absolute inset-0 h-full w-full text-cyan-500/35"
                          preserveAspectRatio="none"
                          viewBox="0 0 16 200"
                        >
                          <motion.path
                            d="M 8 0 L 8 28 L 3 34 L 13 40 L 8 52 L 8 76 L 2 82 L 14 88 L 8 100 L 8 124 L 4 130 L 12 136 L 8 148 L 8 172 L 5 178 L 11 184 L 8 200"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.2"
                            vectorEffect="non-scaling-stroke"
                            initial={{ pathLength: 1, opacity: 0.35 }}
                            animate={{ opacity: [0.32, 0.5, 0.38, 0.32] }}
                            transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
                          />
                        </svg>
                        <div className="absolute bottom-0 left-1/2 top-0 w-px -translate-x-1/2 bg-slate-700/80" />
                      </div>

                      <ul className="relative min-w-0 flex-1 space-y-10">
                        {(selectedRun.bossEncounters ?? []).map((enc, idx) => {
                          const loadoutIds = enc.loadout ?? [];
                          const bossLabel = bossNameById(bossesCatalog, enc.bossId);
                          const bossLifeSec = durationToSeconds(enc.lifespan);
                          const barPct = Math.min(100, Math.round((bossLifeSec / maxBossLifespanSec) * 100));

                          return (
                            <li key={`${selectedRun.id}-enc-${idx}`} className="relative">
                              <p className="font-display pb-3 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                                Encounter {idx + 1}
                              </p>

                              <div className="space-y-4">
                                <div className="overflow-visible pt-2">
                                  <div className="mb-2 flex items-center gap-2">
                                    <Package className="h-3.5 w-3.5 text-cyan-500/70" aria-hidden />
                                    <span className="font-display text-[9px] font-bold uppercase tracking-wider text-cyan-200/80">
                                      Stage 2 · Synergy nodes
                                    </span>
                                  </div>
                                  {loadoutIds.length === 0 ? (
                                    <p className="font-data text-xs text-slate-600">No loadout recorded.</p>
                                  ) : (
                                    <div className="flex flex-wrap gap-3 overflow-visible pb-2">
                                      {loadoutIds.map((itemId) => {
                                        const row = itemById(itemsCatalog, itemId);
                                        const bonus = itemSynergySeconds(row);
                                        return (
                                          <RunItemChip
                                            key={`${selectedRun.id}-enc-${idx}-syn-${itemId}`}
                                            item={row}
                                            itemId={itemId}
                                            run={selectedRun}
                                            synergyBonusSeconds={bonus}
                                          />
                                        );
                                      })}
                                    </div>
                                  )}
                                </div>

                                <div
                                  className="relative overflow-hidden rounded-xl border bg-emerald-500/10 text-slate-200 p-4"
                                  style={{
                                    backgroundImage: `repeating-linear-gradient(
                                      -12deg,
                                      transparent,
                                      transparent 3px,
                                      rgba(239,68,68,0.04) 3px,
                                      rgba(239,68,68,0.04) 4px
                                    )`,
                                  }}
                                >
                                  <div className="relative z-1">
                                    <div className="mb-3 flex items-center gap-2">
                                      <Swords className="h-3.5 w-3.5 text-red-400/90" aria-hidden />
                                      <span className="font-display text-[9px] font-bold uppercase tracking-wider text-red-200/90">
                                        Stage 3 · High-intensity zone
                                      </span>
                                    </div>
                                    <p className="font-data text-sm text-slate-200">
                                      <span className="inline-flex items-center gap-1.5">
                                        <Crosshair className="h-3.5 w-3.5 text-slate-500" aria-hidden />
                                        {bossLabel ?? `Boss ${enc.bossId}`}
                                      </span>
                                      <span className="ml-2 font-mono text-xs text-slate-500">
                                        bossId {enc.bossId}
                                      </span>
                                    </p>
                                    <p className="font-data mt-2 text-[10px] uppercase tracking-wider text-slate-500">
                                      Lifespan (survived)
                                    </p>
                                    <div className="mt-2 h-2 overflow-hidden rounded border border-slate-700 bg-slate-900/80">
                                      <div
                                        className="h-full rounded border-cyan-500/40 transition-[width] duration-500"
                                        style={{ width: `${barPct}%` }}
                                      />
                                    </div>
                                    <p className="font-data mt-1.5 tabular-nums text-xs text-red-200/85">
                                      {enc.lifespan ?? '—'}
                                    </p>

                                    <div className="mt-4 border-t border-red-500/20 pt-4">
                                      <p className="font-display text-[9px] font-bold uppercase tracking-wider text-slate-400">
                                        Gear trace (this fight)
                                      </p>
                                      {loadoutIds.length === 0 ? (
                                        <p className="font-data mt-2 text-xs text-slate-500">No items tagged.</p>
                                      ) : (
                                        <ul className="mt-2 space-y-1.5">
                                          {loadoutIds.map((itemId) => (
                                            <li
                                              key={`${selectedRun.id}-enc-${idx}-gear-${itemId}`}
                                              className="font-data flex flex-wrap items-baseline gap-x-2 text-sm text-slate-300"
                                            >
                                              <span className="tabular-nums text-cyan-400/80">{itemId}</span>
                                              <span className="text-slate-500">
                                                {itemNameById(itemsCatalog, itemId) ?? '—'}
                                              </span>
                                            </li>
                                          ))}
                                        </ul>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.section>
        </motion.div>
      </div>
    </motion.div>
  );
}
