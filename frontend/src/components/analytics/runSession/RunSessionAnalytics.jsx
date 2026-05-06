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
} from 'recharts';
import { durationToSeconds, formatSecondsAsHMS } from '../../../utils/duration';

/**
 * Run Session Analytics — independent from ANALYTICS sub-tabs.
 * Reads `dashboard.runsHistory`, `items`, and `bosses` from props only (no dataStore writes).
 */

function itemNameById(catalog, id) {
  return catalog.find((i) => i.id === id)?.name ?? `#${id}`;
}

function bossNameById(catalog, id) {
  return catalog.find((b) => b.id === id)?.name ?? `Boss ${id}`;
}

export default function RunSessionAnalytics({ data }) {
  const runsHistory = data.dashboard?.runsHistory ?? [];
  const itemsCatalog = data.dashboard?.items ?? [];
  const bossesCatalog = data.dashboard?.bosses ?? [];

  const scatterData = useMemo(
    () =>
      runsHistory.map((run, index) => ({
        order: index + 1,
        durationSec: durationToSeconds(run.duration),
        runId: run.id,
        date: run.date,
        durationLabel: run.duration,
        run,
      })),
    [runsHistory]
  );

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
      if (cx == null || cy == null) return null;
      const active = payload?.runId === selectedRunId;
      const r = active ? 9 : 6;
      const onActivate = (e) => {
        e.stopPropagation();
        if (payload?.runId != null) setSelectedRunId(payload.runId);
      };
      return (
        <g className="cursor-pointer" onClick={onActivate}>
          <circle
            cx={cx}
            cy={cy}
            r={r + 3}
            fill="rgba(34,211,238,0.12)"
            stroke="none"
          />
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill={active ? '#22d3ee' : '#64748b'}
            stroke={active ? '#a5f3fc' : '#475569'}
            strokeWidth={active ? 2 : 1}
            className="transition-[r,fill] duration-150"
            style={{ pointerEvents: 'auto' }}
          />
        </g>
      );
    },
    [selectedRunId]
  );

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
        <p className="font-data mt-2 max-w-2xl text-sm text-slate-500">
          Extracted runs from processed clips. Duration on the vertical axis is a proxy for stability — longer
          sessions read as more successful runs; shorter points suggest higher difficulty or early termination.
        </p>
      </header>

      <div className="flex min-h-[min(640px,72vh)] flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0 lg:rounded-2xl lg:border lg:border-slate-800 lg:bg-slate-950/45 lg:shadow-[inset_0_1px_0_rgba(34,211,238,0.06)]">
        {/* B — Sidebar */}
        <aside className="flex w-full flex-col border-slate-800/80 lg:w-[min(100%,280px)] lg:shrink-0 lg:border-r lg:border-slate-800/90 lg:bg-slate-950/55">
          <div className="border-b border-slate-800/80 px-4 py-3">
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
              Run selection
            </p>
            <p className="font-data mt-1 text-[10px] text-slate-600">
              {runsHistory.length} session{runsHistory.length === 1 ? '' : 's'} in history
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
          {/* A — Scatter */}
          <section className="border-b border-slate-800/80 p-4 backdrop-blur-sm md:p-6">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Activity className="h-4 w-4 text-cyan-400/80" aria-hidden />
              <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-300">
                Success signal · duration vs run order
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
                      name="Run order"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      label={{
                        value: 'Run order',
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
                        value: 'Duration (sec)',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                    />
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
                        return p ? `${p.runId} · order ${p.order}` : '';
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

          {/* C — Detail timeline */}
          <section className="flex-1 p-4 md:p-6">
            <div className="mb-4 flex items-center gap-2">
              <Timer className="h-4 w-4 text-slate-400" aria-hidden />
              <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                Run breakdown
              </h3>
            </div>

            {!selectedRun ? (
              <p className="font-data text-sm text-slate-500">Select a run to inspect internal events.</p>
            ) : (
              <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5 ring-1 ring-cyan-500/10">
                <p className="font-display text-xs font-bold uppercase tracking-widest text-slate-300">
                  {selectedRun.id}
                </p>
                <p className="font-data mt-1 text-xs text-slate-500">
                  {selectedRun.date} · {selectedRun.duration}
                </p>

                <div className="relative mt-6 space-y-8 border-l border-slate-700 pl-6">
                  {(selectedRun.bossEncounters ?? []).map((enc, idx) => (
                    <div key={`${selectedRun.id}-enc-${idx}`} className="relative">
                      <span className="absolute left-[-25px] top-1 h-2.5 w-2.5 rounded-full border-2 border-cyan-400/80 bg-slate-950 shadow-[0_0_8px_rgba(34,211,238,0.35)]" />
                      <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                        Encounter {idx + 1}
                      </p>

                      <div className="mt-3 space-y-3">
                        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                          <div className="mb-2 flex items-center gap-2">
                            <Package className="h-3.5 w-3.5 text-violet-400" aria-hidden />
                            <span className="font-display text-[9px] font-bold uppercase tracking-wider text-violet-200/90">
                              Stage 2 · Choices
                            </span>
                          </div>
                          {(enc.loadout?.length ?? 0) === 0 ? (
                            <p className="font-data text-xs text-slate-600">No loadout recorded.</p>
                          ) : (
                            <ul className="space-y-1">
                              {enc.loadout.map((itemId) => (
                                <li key={`${idx}-item-${itemId}`} className="font-data text-sm text-slate-300">
                                  {itemNameById(itemsCatalog, itemId)}
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>

                        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                          <div className="mb-2 flex items-center gap-2">
                            <Swords className="h-3.5 w-3.5 text-amber-400/90" aria-hidden />
                            <span className="font-display text-[9px] font-bold uppercase tracking-wider text-amber-100/90">
                              Stage 3 · Boss event
                            </span>
                          </div>
                          <p className="font-data text-sm text-slate-200">
                            <span className="inline-flex items-center gap-1.5">
                              <Crosshair className="h-3.5 w-3.5 text-slate-500" aria-hidden />
                              {bossNameById(bossesCatalog, enc.bossId)}
                            </span>
                          </p>
                          <p className="font-data mt-1 text-xs tabular-nums text-slate-500">
                            Lifespan (survived):{' '}
                            <span className="text-slate-300">{enc.lifespan ?? '—'}</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {(selectedRun.bossEncounters ?? []).length === 0 ? (
                  <p className="font-data mt-6 text-sm text-slate-600">No boss encounters for this run.</p>
                ) : null}
              </div>
            )}
          </section>
        </div>
      </div>
    </motion.div>
  );
}
