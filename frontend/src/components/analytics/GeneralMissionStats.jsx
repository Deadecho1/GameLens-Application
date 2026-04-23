import { useMemo, useId } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { motion } from 'framer-motion';
import { Activity, Clock, Hourglass, Package, Crosshair, Sparkles, Swords } from 'lucide-react';
import { useCountUp } from '../../hooks/useCountUp';
import { durationToSeconds, secondsToMinutes, formatSecondsAsHMS } from '../../utils/duration';

/**
 * GENERAL — Run analytics from dashboard.runsHistory, dashboard.bosses, dashboard.items.
 * No action-type or combat/exploration/menu breakdowns.
 */
export default function GeneralMissionStats({ data }) {
  const svgIds = useId().replace(/:/g, '');
  const lineGlowId = `${svgIds}-line-glow`;

  const runsHistory = data.dashboard.runsHistory ?? [];
  const bosses = data.dashboard.bosses ?? [];
  const items = data.dashboard.items ?? [];

  const runMetrics = useMemo(() => {
    const n = runsHistory.length;
    if (n === 0) return { totalRuns: 0, avgSec: 0, longestSec: 0 };
    let sum = 0;
    let longestSec = 0;
    for (const run of runsHistory) {
      const sec = durationToSeconds(run.duration);
      sum += sec;
      if (sec > longestSec) longestSec = sec;
    }
    return { totalRuns: n, avgSec: sum / n, longestSec };
  }, [runsHistory]);

  const totalRunsAnimated = useCountUp(runMetrics.totalRuns, 1400);
  const totalItemsAnimated = useCountUp(items.length, 1400);

  const chartData = useMemo(
    () =>
      runsHistory.map((run) => ({
        runId: run.id,
        date: run.date,
        minutes: secondsToMinutes(durationToSeconds(run.duration)),
        durationRaw: run.duration,
      })),
    [runsHistory]
  );

  const sortedRunsDesc = useMemo(() => {
    return [...runsHistory]
      .map((run) => ({
        ...run,
        sec: durationToSeconds(run.duration),
      }))
      .sort((a, b) => b.sec - a.sec);
  }, [runsHistory]);

  const mostPopularItem = useMemo(() => {
    if (!items.length) return null;
    return items.reduce((best, it) =>
      (it.popularity ?? 0) > (best.popularity ?? 0) ? it : best
    );
  }, [items]);

  const bossKill = useMemo(() => {
    const total = bosses.length;
    if (!total) return { percent: 0, defeated: 0, total: 0 };
    const defeated = bosses.filter(
      (b) => String(b.status ?? '').toLowerCase() === 'defeated'
    ).length;
    return {
      percent: Math.round((defeated / total) * 100),
      defeated,
      total,
    };
  }, [bosses]);

  const killRateAnimated = useCountUp(bossKill.percent, 1600);

  const avgLabel = formatSecondsAsHMS(Math.round(runMetrics.avgSec));
  const longestLabel = formatSecondsAsHMS(Math.round(runMetrics.longestSec));

  return (
    <div className="space-y-8">
      <header>
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          General analytics
        </p>
        <h3 className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl">
          Run command deck
        </h3>

      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <BriefMetricCard title="Total runs" subtitle="All recorded sessions" icon={Activity} accent="cyan" hero>
          <motion.p
            className="font-data text-2xl font-bold tabular-nums tracking-tight text-cyan-200 md:text-3xl"
            animate={{
              textShadow: [
                '0 0 20px rgba(34,211,238,0.35)',
                '0 0 32px rgba(34,211,238,0.55)',
                '0 0 20px rgba(34,211,238,0.35)',
              ],
            }}
            transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
          >
            {totalRunsAnimated}
          </motion.p>
        </BriefMetricCard>

        <BriefMetricCard title="Avg. run time" subtitle="Mean session duration" icon={Clock} accent="blue">
          <p className="font-data text-2xl font-bold tabular-nums text-slate-100 md:text-3xl">{avgLabel}</p>
        </BriefMetricCard>

        <BriefMetricCard title="Longest session" subtitle="Max duration from history" icon={Hourglass} accent="blue">
          <p className="font-data text-2xl font-bold tabular-nums text-slate-100 md:text-3xl">{longestLabel}</p>
        </BriefMetricCard>

        <BriefMetricCard
          title="Total items found"
          subtitle="Unique items tracked (catalog)"
          icon={Package}
          accent="slate"
        >
          <p className="font-data text-3xl font-bold tabular-nums text-slate-100 md:text-4xl">
            {totalItemsAnimated}
          </p>
        </BriefMetricCard>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <div className="rounded-2xl border border-slate-800 bg-transparent p-4 backdrop-blur-md md:p-6">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
              <h4 className="font-display text-xs font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Run duration trend
              </h4>
              <span className="font-data text-[10px] text-slate-600">Run ID · minutes</span>
            </div>
            <div className="h-[320px] w-full min-w-0">
              {chartData.length === 0 ? (
                <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">
                  <p className="font-data text-sm text-slate-500">No runs recorded yet.</p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
                    <defs>
                      <filter id={lineGlowId} x="-40%" y="-40%" width="180%" height="180%">
                        <feGaussianBlur stdDeviation="2" result="blur" />
                        <feMerge>
                          <feMergeNode in="blur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>
                    </defs>
                    <CartesianGrid strokeDasharray="3 6" stroke="#334155" strokeOpacity={0.6} vertical={false} />
                    <XAxis
                      dataKey="runId"
                      name="Run ID"
                      tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      label={{
                        value: 'Run ID',
                        position: 'insideBottom',
                        offset: -2,
                        fill: '#64748b',
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                    />
                    <YAxis
                      tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      tickFormatter={(v) => `${v}`}
                      label={{
                        value: 'Minutes',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#64748b',
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                    />
                    <Tooltip
                      cursor={{ stroke: '#475569', strokeDasharray: '4 4' }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const row = payload[0].payload;
                        const sec = durationToSeconds(row.durationRaw);
                        return (
                          <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                            <p className="font-data text-xs font-semibold text-cyan-200">{row.runId}</p>
                            <p className="font-data text-[10px] text-slate-500">{row.date}</p>
                            <p className="font-data mt-1 text-sm text-slate-200">
                              Duration:{' '}
                              <span className="tabular-nums text-blue-300">{row.durationRaw}</span>
                            </p>
                            <p className="font-data text-[10px] text-slate-500">
                              {formatSecondsAsHMS(sec)} · {row.minutes} min
                            </p>
                          </div>
                        );
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="minutes"
                      name="Duration (min)"
                      stroke="#22d3ee"
                      strokeWidth={2.5}
                      filter={`url(#${lineGlowId})`}
                      dot={{ r: 3, fill: '#a5f3fc', stroke: '#0891b2', strokeWidth: 1 }}
                      activeDot={{ r: 5, fill: '#ecfeff', stroke: '#22d3ee', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-cyan-500/20 bg-slate-900/30 p-4 ring-1 ring-cyan-500/15 backdrop-blur-md md:p-5">
            <div className="mb-3 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-cyan-400" strokeWidth={1.25} aria-hidden />
              <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-300/90">
                Highlight
              </h4>
            </div>
            {mostPopularItem ? (
              <>
                <p className="font-display text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Most popular item
                </p>
                <p className="font-data mt-2 text-lg font-bold text-slate-100">{mostPopularItem.name}</p>
                <p className="font-data mt-1 text-sm tabular-nums text-cyan-300/90">
                  Popularity <span className="text-cyan-200">{mostPopularItem.popularity}</span>
                </p>
              </>
            ) : (
              <p className="font-data text-sm text-slate-500">No item catalog data.</p>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/25 p-4 backdrop-blur-md ring-1 ring-blue-500/15 md:p-5">
            <div className="mb-3 flex items-center gap-2">
              <Swords className="h-4 w-4 text-blue-400" strokeWidth={1.25} aria-hidden />
              <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Boss kill rate
              </h4>
            </div>
            {bossKill.total === 0 ? (
              <p className="font-data text-sm text-slate-500">No boss encounters logged.</p>
            ) : (
              <>
                <div className="flex items-baseline gap-2">
                  <span className="font-data text-4xl font-bold tabular-nums text-cyan-200">{killRateAnimated}</span>
                  <span className="font-data text-xl font-semibold text-blue-400/80">%</span>
                </div>
                <p className="font-data mt-2 text-[11px] text-slate-500">
                  <span className="tabular-nums text-slate-400">{bossKill.defeated}</span> defeated ·{' '}
                  <span className="tabular-nums text-slate-400">{bossKill.total}</span> encountered
                </p>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
                  <motion.div
                    className="h-full rounded-full bg-linear-to-r from-cyan-500 to-blue-500 shadow-[0_0_12px_rgba(34,211,238,0.45)]"
                    initial={{ width: 0 }}
                    animate={{ width: `${bossKill.percent}%` }}
                    transition={{ duration: 1.2, ease: 'easeOut' }}
                  />
                </div>
              </>
            )}
          </div>

          <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-slate-800 bg-transparent p-4 backdrop-blur-md md:p-5">
            <div className="mb-3 flex items-center gap-2">
              <Crosshair className="h-4 w-4 text-slate-400" strokeWidth={1.25} aria-hidden />
              <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Performance
              </h4>
            </div>
            <p className="font-data mb-2 text-[10px] text-slate-600">Sorted longest → shortest</p>
            <ul className="font-data max-h-[220px] min-h-[120px] space-y-2 overflow-y-auto pr-1 text-[11px] [scrollbar-color:rgba(51,65,85,0.9)_transparent]">
              {sortedRunsDesc.length === 0 ? (
                <li className="py-6 text-center text-slate-500">No run history.</li>
              ) : (
                sortedRunsDesc.map((run) => (
                  <li
                    key={run.id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-slate-800/80 bg-slate-950/40 px-3 py-2"
                  >
                    <span className="font-medium text-cyan-200/90">{run.id}</span>
                    <span className="shrink-0 tabular-nums text-slate-300">{run.duration}</span>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function BriefMetricCard({ title, subtitle, icon: Icon, children, accent, hero = false, className = '' }) {
  const ring = {
    cyan: 'ring-cyan-500/20',
    blue: 'ring-blue-500/20',
    emerald: 'ring-emerald-500/20',
    slate: 'ring-slate-600/30',
  };
  const iconCls = {
    cyan: 'text-cyan-400',
    blue: 'text-blue-400',
    emerald: 'text-emerald-400',
    slate: 'text-slate-400',
  };

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/25 p-5 backdrop-blur-md ring-1 ${ring[accent]} ${className}`}
    >
      {hero && (
        <motion.div
          className="pointer-events-none absolute inset-0 rounded-2xl border border-cyan-500/20"
          animate={{ opacity: [0.35, 0.65, 0.35] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}
      <div className="relative mb-3 flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <Icon className={`h-5 w-5 ${iconCls[accent]}`} strokeWidth={1.25} />
            <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">
              {title}
            </h3>
          </div>
          <p className="font-data mt-1 text-[10px] text-slate-600">{subtitle}</p>
        </div>
      </div>
      <div className="relative font-data">{children}</div>
    </div>
  );
}
