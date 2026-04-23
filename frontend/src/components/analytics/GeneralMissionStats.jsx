import { useMemo, useId } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import { Activity, Clock, TrendingUp, Layers } from 'lucide-react';
import { useCountUp } from '../../hooks/useCountUp';
import { durationToSeconds, secondsToMinutes, formatSecondsAsHMS } from '../../utils/duration';

/**
 * GENERAL — System Mission Briefing.
 * Total time = sum(dashboard.runsHistory[].duration); counts & scores from dashboard.stats;
 * distribution from dashboard.actionTypeDistribution.
 */
export default function GeneralMissionStats({ data }) {
  const svgIds = useId().replace(/:/g, '');
  const areaFillId = `${svgIds}-area-fill`;
  const areaGlowId = `${svgIds}-area-glow`;

  const { stats, runsHistory, actionTypeDistribution } = data.dashboard;

  const totalSecondsFromHistory = useMemo(
    () => runsHistory.reduce((acc, run) => acc + durationToSeconds(run.duration), 0),
    [runsHistory]
  );

  const totalTimeAnimated = useCountUp(totalSecondsFromHistory, 1800);
  const analysisCountAnimated = useCountUp(stats.totalRuns, 1400);
  const efficiencyAnimated = useCountUp(stats.efficiencyScore, 1600);

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

  const pieData = useMemo(
    () =>
      (actionTypeDistribution || []).map((row) => ({
        name: row.name,
        value: row.value,
        fill: row.fill,
      })),
    [actionTypeDistribution]
  );

  return (
    <div className="space-y-8">
      <header>
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          System mission briefing
        </p>
        <h3 className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl">
          Command dashboard
        </h3>
        <p className="font-data mt-2 text-sm text-slate-500">
          Aggregate time from <code className="text-cyan-700/90">runsHistory</code> · KPIs from{' '}
          <code className="text-cyan-700/90">dashboard.stats</code>
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <BriefMetricCard
          title="Total game time"
          subtitle="Sum of session durations"
          icon={Clock}
          accent="cyan"
          hero
        >
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
            {formatSecondsAsHMS(totalTimeAnimated)}
          </motion.p>
        </BriefMetricCard>

        <BriefMetricCard
          title="Analysis count"
          subtitle="Total runs recorded"
          icon={Activity}
          accent="blue"
        >
          <p className="font-data text-3xl font-bold tabular-nums text-slate-100 md:text-4xl">
            {analysisCountAnimated}
          </p>
        </BriefMetricCard>

        <BriefMetricCard
          title="Efficiency score"
          subtitle="Model confidence (mock)"
          icon={TrendingUp}
          accent="emerald"
        >
          <div className="flex items-baseline gap-2">
            <p className="font-data text-3xl font-bold tabular-nums text-emerald-300 md:text-4xl">
              {efficiencyAnimated}
            </p>
            <span className="font-data text-lg font-semibold text-emerald-500/80">%</span>
            <TrendingUp className="mb-1 ml-1 h-5 w-5 text-emerald-400" aria-hidden />
          </div>
        </BriefMetricCard>

        <BriefMetricCard
          title="System health"
          subtitle="Fleet posture"
          icon={Layers}
          accent="slate"
        >
          <p className="font-data text-xl font-bold uppercase tracking-wider text-cyan-200 md:text-2xl">
            {stats.systemHealthLabel}
          </p>
          <div className="mt-3 flex items-center gap-2">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
            <span className="font-data text-[10px] uppercase tracking-wider text-slate-500">
              Nominal
            </span>
          </div>
        </BriefMetricCard>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <div className="rounded-2xl border border-slate-800 bg-transparent p-4 backdrop-blur-md md:p-6">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
              <h4 className="font-display text-xs font-bold uppercase tracking-[0.2em] text-blue-400/90">
                Session duration trends
              </h4>
              <span className="font-data text-[10px] text-slate-600">Run ID · minutes</span>
            </div>
            <div className="h-[320px] w-full min-w-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
                  <defs>
                    <linearGradient id={areaFillId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#1d4ed8" stopOpacity={0.55} />
                      <stop offset="45%" stopColor="#1e3a8a" stopOpacity={0.25} />
                      <stop offset="100%" stopColor="#0f172a" stopOpacity={0} />
                    </linearGradient>
                    <filter id={areaGlowId} x="-30%" y="-30%" width="160%" height="160%">
                      <feGaussianBlur stdDeviation="2.5" result="blur" />
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
                      value: 'Duration (min)',
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
                  <Area
                    type="monotone"
                    dataKey="minutes"
                    name="Duration (min)"
                    stroke="#22d3ee"
                    strokeWidth={2.5}
                    fill={`url(#${areaFillId})`}
                    filter={`url(#${areaGlowId})`}
                    dot={{ r: 3, fill: '#a5f3fc', stroke: '#0891b2', strokeWidth: 1 }}
                    activeDot={{ r: 5, fill: '#ecfeff', stroke: '#22d3ee', strokeWidth: 2 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-transparent p-4 backdrop-blur-md md:p-6">
          <h4 className="font-display mb-1 text-xs font-bold uppercase tracking-[0.2em] text-blue-400/90">
            Action type distribution
          </h4>
          <p className="font-data mb-4 text-[10px] text-slate-600">Mock mix · dashboard.actionTypeDistribution</p>
          <div className="h-[280px] w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={58}
                  outerRadius={88}
                  paddingAngle={2}
                  stroke="none"
                >
                  {pieData.map((entry, i) => (
                    <Cell key={`cell-${entry.name}-${i}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const p = payload[0];
                    return (
                      <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 backdrop-blur-md">
                        <p className="font-data text-xs font-semibold text-slate-200">{p.name}</p>
                        <p className="font-data tabular-nums text-sm text-cyan-300">{p.value}%</p>
                      </div>
                    );
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <ul className="font-data mt-2 space-y-1.5 text-[11px] text-slate-400">
            {pieData.map((row) => (
              <li key={row.name} className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: row.fill }} />
                  {row.name}
                </span>
                <span className="tabular-nums text-slate-300">{row.value}%</span>
              </li>
            ))}
          </ul>
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
