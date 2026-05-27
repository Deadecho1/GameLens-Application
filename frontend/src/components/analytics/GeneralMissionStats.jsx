import { useMemo } from 'react';
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
import { motion } from 'framer-motion';
import { Activity, Clock, Hourglass, Package, Crosshair, Sparkles, Swords } from 'lucide-react';
import { useCountUp } from '../../hooks/useCountUp';
import { durationToSeconds, formatSecondsAsHMS } from '../../utils/duration';
import { buildRunDurationDistribution } from '../../utils/runDurationDistribution';
import { computeGeneralMetrics } from '../../utils/analyticsGeneralMetrics';
import DeltaIndicator from './DeltaIndicator';

/**
 * GENERAL — Run analytics from dashboard.runsHistory, dashboard.bosses, dashboard.items.
 * No action-type or combat/exploration/menu breakdowns.
 */
const HISTOGRAM_BAR_FILL = '#22d3ee';

export default function GeneralMissionStats({ data, compareBaseline = null }) {
  const runsHistory = data.dashboard.runsHistory ?? [];
  const bosses = data.dashboard.bosses ?? [];
  const items = data.dashboard.items ?? [];

  const metrics = useMemo(() => computeGeneralMetrics(data), [data]);
  const baselineMetrics = useMemo(
    () => (compareBaseline ? computeGeneralMetrics(compareBaseline) : null),
    [compareBaseline],
  );

  const runMetrics = useMemo(
    () => ({
      totalRuns: metrics.totalRuns,
      avgSec: metrics.avgSec,
      longestSec: metrics.longestSec,
    }),
    [metrics],
  );

  const totalRunsAnimated = useCountUp(runMetrics.totalRuns, 1400);
  const totalItemsAnimated = useCountUp(items.length, 1400);

  const histogramData = useMemo(
    () => buildRunDurationDistribution(runsHistory),
    [runsHistory],
  );

  const histogramTotal = useMemo(
    () => histogramData.reduce((sum, row) => sum + row.count, 0),
    [histogramData],
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

  const bossKill = useMemo(
    () => ({
      percent: metrics.bossKillPercent,
      defeated: bosses.filter(
        (b) => String(b.status ?? '').toLowerCase() === 'defeated',
      ).length,
      total: bosses.length,
    }),
    [bosses, metrics.bossKillPercent],
  );

  const killRateAnimated = useCountUp(bossKill.percent, 1600);

  const avgLabel = formatSecondsAsHMS(Math.round(runMetrics.avgSec));
  const longestLabel = formatSecondsAsHMS(Math.round(runMetrics.longestSec));

  return (
    <div className="space-y-8">
      <header>
        <p className="font-display text-sm font-bold uppercase tracking-[0.35em] text-blue-500/70">
          General analytics
        </p>
        <h3 className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl">
          Run summary
        </h3>

      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <BriefMetricCard title="Total runs" subtitle="All recorded runs" icon={Activity} accent="cyan" hero>
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
          {baselineMetrics ? (
            <DeltaIndicator
              kind="count"
              baseline={baselineMetrics.totalRuns}
              current={metrics.totalRuns}
            />
          ) : null}
        </BriefMetricCard>

        <BriefMetricCard title="Avg. run time" subtitle="Average run time" icon={Clock} accent="blue">
          <p className="font-data text-2xl font-bold tabular-nums text-slate-100 md:text-3xl">{avgLabel}</p>
          {baselineMetrics ? (
            <DeltaIndicator
              kind="duration"
              baseline={baselineMetrics.avgSec}
              current={metrics.avgSec}
            />
          ) : null}
        </BriefMetricCard>

        <BriefMetricCard title="Longest run" subtitle="Maximum run time" icon={Hourglass} accent="blue">
          <p className="font-data text-2xl font-bold tabular-nums text-slate-100 md:text-3xl">{longestLabel}</p>
          {baselineMetrics ? (
            <DeltaIndicator
              kind="duration"
              baseline={baselineMetrics.longestSec}
              current={metrics.longestSec}
            />
          ) : null}
        </BriefMetricCard>

        <BriefMetricCard
          title="Total items found"
          subtitle="Unique items tracked"
          icon={Package}
          accent="slate"
        >
          <p className="font-data text-3xl font-bold tabular-nums text-slate-100 md:text-4xl">
            {totalItemsAnimated}
          </p>
          {baselineMetrics ? (
            <DeltaIndicator
              kind="count"
              baseline={baselineMetrics.totalItems}
              current={metrics.totalItems}
            />
          ) : null}
        </BriefMetricCard>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <div className="rounded-2xl border border-slate-800 bg-transparent p-4 backdrop-blur-md md:p-6">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
              <h4 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Run time distribution
              </h4>
              <span className="font-data text-base text-slate-300">
                {histogramTotal} run{histogramTotal === 1 ? '' : 's'} · 5 min intervals
              </span>
            </div>
            <div className="h-[320px] w-full min-w-0">
              {histogramTotal === 0 ? (
                <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">
                  <p className="font-data text-base text-slate-300">No runs recorded yet.</p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histogramData} margin={{ top: 8, right: 12, left: 4, bottom: 28 }}>
                    <CartesianGrid strokeDasharray="3 6" stroke="#334155" strokeOpacity={0.6} vertical={false} />
                    <XAxis
                      dataKey="bucket"
                      tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      interval={0}
                      angle={-22}
                      textAnchor="end"
                      height={52}
                      label={{
                        value: 'Time interval',
                        position: 'insideBottom',
                        offset: -4,
                        fill: '#64748b',
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      label={{
                        value: 'Number of runs',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#64748b',
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                    />
                    <Tooltip
                      cursor={{ fill: 'rgba(34,211,238,0.08)' }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const row = payload[0].payload;
                        const pct =
                          histogramTotal > 0
                            ? Math.round((row.count / histogramTotal) * 100)
                            : 0;
                        return (
                          <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                            <p className="font-data text-sm font-semibold text-cyan-200">{row.bucket}</p>
                            <p className="font-data mt-1 text-sm text-slate-200">
                              <span className="tabular-nums text-blue-300">{row.count}</span> run
                              {row.count === 1 ? '' : 's'}
                            </p>
                            <p className="font-data text-base text-slate-300">{pct}% of runs</p>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="count" name="Runs" radius={[6, 6, 0, 0]} maxBarSize={48}>
                      {histogramData.map((entry) => (
                        <Cell
                          key={entry.bucket}
                          fill={entry.count > 0 ? HISTOGRAM_BAR_FILL : 'rgba(51,65,85,0.45)'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-cyan-500/20 bg-slate-900/30 p-4 ring-1 ring-cyan-500/15 backdrop-blur-md md:p-5">
            <div className="mb-3 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-cyan-400" strokeWidth={1.25} aria-hidden />
              <h4 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-cyan-300/90">
                Top item
              </h4>
            </div>
            {mostPopularItem ? (
              <>
                <p className="font-display text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Most used item
                </p>
                <p className="font-data mt-2 text-lg font-bold text-slate-100">{mostPopularItem.name}</p>
                <p className="font-data mt-1 text-sm tabular-nums text-cyan-300/90">
                  Pick rate:{' '}
                  <span className="text-cyan-200">
                    {Math.round(Number(mostPopularItem.popularity))}%
                  </span>
                </p>
                {baselineMetrics &&
                metrics.mostPopularPopularity != null &&
                baselineMetrics.mostPopularPopularity != null ? (
                  <DeltaIndicator
                    kind="percent"
                    baseline={baselineMetrics.mostPopularPopularity}
                    current={metrics.mostPopularPopularity}
                  />
                ) : null}
              </>
            ) : (
              <p className="font-data text-base text-slate-300">No item catalog data.</p>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/25 p-4 backdrop-blur-md ring-1 ring-blue-500/15 md:p-5">
            <div className="mb-3 flex items-center gap-2">
              <Swords className="h-4 w-4 text-blue-400" strokeWidth={1.25} aria-hidden />
              <h4 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Boss success rate
              </h4>
            </div>
            {bossKill.total === 0 ? (
              <p className="font-data text-base text-slate-300">No boss fights logged.</p>
            ) : (
              <>
                <div className="flex items-baseline gap-2">
                  <span className="font-data text-4xl font-bold tabular-nums text-cyan-200">{killRateAnimated}</span>
                  <span className="font-data text-xl font-semibold text-blue-400/80">%</span>
                </div>
                {baselineMetrics ? (
                  <DeltaIndicator
                    kind="percent"
                    baseline={baselineMetrics.bossKillPercent}
                    current={metrics.bossKillPercent}
                  />
                ) : null}
                <p className="font-data mt-2 text-base text-slate-300">
                  <span className="tabular-nums text-slate-300">{bossKill.defeated}</span> defeated ·{' '}
                  <span className="tabular-nums text-slate-300">{bossKill.total}</span> Boss Fights
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
              <Crosshair className="h-4 w-4 text-slate-300" strokeWidth={1.25} aria-hidden />
              <h4 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Run history
              </h4>
            </div>
            <p className="font-data mb-2 text-base text-slate-300">Sorted by longest time</p>
            <ul className="font-data max-h-[220px] min-h-[120px] space-y-2 overflow-y-auto pr-1 text-sm [scrollbar-color:rgba(51,65,85,0.9)_transparent]">
              {sortedRunsDesc.length === 0 ? (
                <li className="py-6 text-center text-slate-300">No run history.</li>
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
    slate: 'text-slate-300',
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
            <h3 className="font-display text-sm font-bold uppercase tracking-[0.18em] text-slate-300">
              {title}
            </h3>
          </div>
          <p className="font-data mt-1 text-base text-slate-300">{subtitle}</p>
        </div>
      </div>
      <div className="relative font-data">{children}</div>
    </div>
  );
}
