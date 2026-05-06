import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Calendar,
  Clock,
  Crosshair,
  Package,
  Swords,
  Timer,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ScatterChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Scatter,
  ReferenceLine,
} from 'recharts';
import { initialData } from '../../../dataStore.js';
import { durationToSeconds, formatSecondsAsHMS } from '../../../utils/duration';

/**
 * Converts run `duration` strings from dataStore (e.g. "00:28:00") to seconds for charts.
 */
export function runDurationToSeconds(hms) {
  return durationToSeconds(hms);
}

function itemNameById(catalog, id) {
  return catalog.find((i) => i.id === id)?.name ?? null;
}

function bossNameById(catalog, id) {
  return catalog.find((b) => b.id === id)?.name ?? null;
}

/** Longer runs → stronger "success" tint (cyan/emerald); shorter → cooler slate. */
function successFillForDuration(durationSec, minSec, maxSec, isSelected) {
  const spread = maxSec - minSec || 1;
  const t = Math.min(1, Math.max(0, (durationSec - minSec) / spread));
  if (isSelected) return '#22d3ee';
  if (t >= 0.66) return '#34d399';
  if (t >= 0.33) return '#5eead4';
  return '#94a3b8';
}

function successStrokeForDuration(durationSec, minSec, maxSec, isSelected) {
  if (isSelected) return '#a5f3fc';
  const spread = maxSec - minSec || 1;
  const t = Math.min(1, Math.max(0, (durationSec - minSec) / spread));
  if (t >= 0.66) return '#6ee7b7';
  if (t >= 0.33) return '#99f6e4';
  return '#64748b';
}

/**
 * Run Session Analytics — runs from live `data` when present; otherwise `initialData.dashboard.runsHistory`.
 * Item and boss labels always resolve from `initialData.dashboard` (dataStore contract).
 */
export default function RunSessionAnalytics({ data }) {
  const runsHistory = useMemo(() => {
    const fallbackRuns = initialData.dashboard.runsHistory ?? [];
    if (!data?.dashboard) return fallbackRuns;
    const runs = data.dashboard.runsHistory;
    if (!Array.isArray(runs) || runs.length === 0) return fallbackRuns;
    return runs;
  }, [data]);

  const itemsCatalog = initialData.dashboard.items ?? [];
  const bossesCatalog = initialData.dashboard.bosses ?? [];

  console.log("Current Runs History:", runsHistory);

  const globalAverageDurationSeconds = useMemo(() => {
    if (!runsHistory.length) return 0;
    const sum = runsHistory.reduce((acc, r) => acc + runDurationToSeconds(r.duration), 0);
    return sum / runsHistory.length;
  }, [runsHistory]);

  const scatterData = useMemo(() => {
    if (!runsHistory.length) return [];
    const secs = runsHistory.map((r) => runDurationToSeconds(r.duration));
    const minSec = Math.min(...secs);
    const maxSec = Math.max(...secs);
    return runsHistory.map((run, index) => {
      const durationSec = secs[index];
      return {
        order: index + 1,
        durationSec,
        runId: run.id,
        date: run.date,
        durationLabel: run.duration,
        run,
        minSec,
        maxSec,
      };
    });
  }, [runsHistory]);

  const [selectedRunId, setSelectedRunId] = useState(null);

  useEffect(() => {
    if (runsHistory.length === 0) {
      setSelectedRunId(null);
      return;
    }
    setSelectedRunId((prev) => {
      if (prev != null && runsHistory.some((r) => r.id === prev)) return prev;
      return runsHistory[0].id;
    });
  }, [runsHistory]);

  const selectedRun = useMemo(
    () => runsHistory.find((r) => r.id === selectedRunId) ?? null,
    [runsHistory, selectedRunId]
  );

  const handleListSelect = useCallback((runId) => {
    setSelectedRunId(runId);
  }, []);

  const CustomScatterShape = useCallback(
    (props) => {
      const { cx, cy, payload } = props;
      if (cx == null || cy == null || !payload) return null;
      const active = payload.runId === selectedRunId;
      const r = active ? 9 : 6;
      const fill = successFillForDuration(payload.durationSec, payload.minSec, payload.maxSec, active);
      const stroke = successStrokeForDuration(payload.durationSec, payload.minSec, payload.maxSec, active);
      const onActivate = (e) => {
        e.stopPropagation();
        if (payload.runId != null) setSelectedRunId(payload.runId);
      };
      return (
        <g className="cursor-pointer" onClick={onActivate}>
          <circle cx={cx} cy={cy} r={r + 3} fill={`${fill}22`} stroke="none" />
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill={fill}
            stroke={stroke}
            strokeWidth={active ? 2 : 1}
            className="transition-[r,fill] duration-150"
            style={{ pointerEvents: 'auto' }}
          />
        </g>
      );
    },
    [selectedRunId]
  );

  const avgLabel = formatSecondsAsHMS(Math.round(globalAverageDurationSeconds));

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className="mx-auto max-w-[1800px] px-4 py-8 md:py-10"
    >
      <header className="mb-6">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-cyan-500/70">
          Session intel
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Run session analytics
        </h2>

      </header>

      <div className="flex min-h-[min(640px,72vh)] flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0 lg:rounded-2xl lg:border lg:border-slate-800 lg:bg-slate-950/45 lg:shadow-[inset_0_1px_0_rgba(34,211,238,0.06)]">
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
                            ? 'border-cyan-500/45 bg-cyan-500/10 shadow-[0_0_16px_rgba(34,211,238,0.08)]'
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

        <div className="flex min-w-0 flex-1 flex-col">
          <section className="border-b border-slate-800/80 p-4 backdrop-blur-sm md:p-6">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Activity className="h-4 w-4 text-cyan-400/80" aria-hidden />
              <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-300">
                Success signal · duration vs run index
              </h3>
            </div>
            {scatterData.length === 0 ? (
              <div className="flex h-[280px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40 font-data text-sm text-slate-500">
                No data to plot.
              </div>
            ) : (
              <div className="h-[min(320px,40vh)] w-full min-w-0 md:h-[360px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 12, right: 16, bottom: 12, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.55} />
                    <XAxis
                      type="number"
                      dataKey="order"
                      name="Run index"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      label={{
                        value: 'Run # (order)',
                        position: 'insideBottom',
                        offset: -4,
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                      allowDecimals={false}
                    />
                    <YAxis
                      type="number"
                      dataKey="durationSec"
                      name="Duration"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      tickFormatter={(v) => `${Math.round(v / 60)}m`}
                      label={{
                        value: 'Duration (seconds)',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                    />
                    {globalAverageDurationSeconds > 0 ? (
                      <ReferenceLine
                        y={globalAverageDurationSeconds}
                        stroke="#fbbf24"
                        strokeDasharray="6 5"
                        strokeOpacity={0.9}
                        label={{
                          value: `Global avg · ${avgLabel}`,
                          position: 'insideTopRight',
                          fill: '#fbbf24',
                          fontSize: 10,
                          fontFamily: 'JetBrains Mono, monospace',
                        }}
                      />
                    ) : null}
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3', stroke: '#64748b' }}
                      contentStyle={{
                        background: 'rgba(15,23,42,0.95)',
                        border: '1px solid #334155',
                        borderRadius: 8,
                        fontSize: 12,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                      formatter={(_, name, item) => {
                        if (name === 'durationSec') {
                          const sec = item?.payload?.durationSec ?? 0;
                          return [formatSecondsAsHMS(sec), 'Duration'];
                        }
                        return [_, name];
                      }}
                      labelFormatter={(_, items) => {
                        const p = items?.[0]?.payload;
                        return p ? `${p.runId} · #${p.order}` : '';
                      }}
                    />
                    <Scatter
                      name="Runs"
                      data={scatterData}
                      fill="#22d3ee"
                      shape={CustomScatterShape}
                      isAnimationActive={false}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>

          <section className="flex-1 p-4 md:p-6">
            <div className="mb-4 flex items-center gap-2">
              <Timer className="h-4 w-4 text-slate-400" aria-hidden />
              <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                Run breakdown
              </h3>
            </div>

            {!selectedRun ? (
              <p className="font-data text-sm text-slate-500">Select a run from the list or chart.</p>
            ) : (
              <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5 ring-1 ring-cyan-500/10">
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

                <h4 className="font-display mt-6 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  Boss encounters
                </h4>

                {(selectedRun.bossEncounters ?? []).length === 0 ? (
                  <p className="font-data mt-3 text-sm text-slate-600">No boss encounters for this run.</p>
                ) : (
                  <ul className="mt-4 space-y-4">
                    {(selectedRun.bossEncounters ?? []).map((enc, idx) => {
                      const loadoutIds = enc.loadout ?? [];
                      const bossLabel = bossNameById(bossesCatalog, enc.bossId);
                      return (
                        <li
                          key={`${selectedRun.id}-enc-${idx}`}
                          className="rounded-xl border border-slate-800 bg-slate-900/35 p-4"
                        >
                          <p className="font-display text-[9px] font-bold uppercase tracking-wider text-slate-500">
                            Encounter {idx + 1}
                          </p>

                          <div className="mt-3 space-y-3 border-l border-slate-700 pl-4">
                            <div>
                              <p className="font-data text-[10px] uppercase tracking-wider text-slate-500">
                                bossId
                              </p>
                              <p className="font-data mt-0.5 text-sm tabular-nums text-slate-200">{enc.bossId}</p>
                              {bossLabel ? (
                                <p className="font-data text-xs text-slate-500">{bossLabel}</p>
                              ) : null}
                            </div>

                            <div>
                              <p className="font-data text-[10px] uppercase tracking-wider text-slate-500">
                                Lifespan (boss survived)
                              </p>
                              <p className="font-data mt-0.5 text-sm tabular-nums text-amber-200/90">
                                {enc.lifespan ?? '—'}
                              </p>
                            </div>

                            <div>
                              <div className="mb-1 flex items-center gap-2">
                                <Package className="h-3.5 w-3.5 text-violet-400" aria-hidden />
                                <p className="font-display text-[9px] font-bold uppercase tracking-wider text-violet-200/90">
                                  Stage 2 · Loadout (item IDs)
                                </p>
                              </div>
                              {loadoutIds.length === 0 ? (
                                <p className="font-data text-xs text-slate-600">None</p>
                              ) : (
                                <ul className="mt-1 space-y-1">
                                  {loadoutIds.map((itemId) => {
                                    const nm = itemNameById(itemsCatalog, itemId);
                                    return (
                                      <li
                                        key={`${selectedRun.id}-enc-${idx}-id-${itemId}`}
                                        className="font-data text-sm text-slate-300"
                                      >
                                        <span className="tabular-nums text-cyan-300/80">id {itemId}</span>
                                        {nm ? (
                                          <span className="text-slate-500"> · {nm}</span>
                                        ) : null}
                                      </li>
                                    );
                                  })}
                                </ul>
                              )}
                            </div>

                            <div className="flex items-start gap-2 border-t border-slate-800 pt-3">
                              <Swords className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400/90" aria-hidden />
                              <div>
                                <p className="font-display text-[9px] font-bold uppercase tracking-wider text-amber-100/90">
                                  Stage 3 · Boss event
                                </p>
                                <p className="font-data mt-1 text-sm text-slate-200">
                                  <span className="inline-flex items-center gap-1.5">
                                    <Crosshair className="h-3.5 w-3.5 text-slate-500" aria-hidden />
                                    {bossLabel ?? `Boss ${enc.bossId}`}
                                  </span>
                                </p>
                              </div>
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </motion.div>
  );
}
