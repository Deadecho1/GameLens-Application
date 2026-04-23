import { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { motion } from 'framer-motion';
import { Radar, Trophy, Timer, Activity, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { useCountUp } from '../../hooks/useCountUp';
import { durationToSeconds, secondsToMinutes, formatSecondsAsHMS } from '../../utils/duration';

/**
 * GENERAL sub-tab — Command Center grid + Run Duration History chart.
 * Data: dashboard.stats, dashboard.runsHistory, processing.status (live), dashboard.stats.lastRunSuccessful.
 */
export default function GeneralMissionStats({ data }) {
  const { stats, runsHistory } = data.dashboard;
  const { processing } = data;

  const totalRunsAnimated = useCountUp(stats.totalRuns, 1500);

  const chartData = useMemo(
    () =>
      runsHistory.map((run) => ({
        label: run.id,
        subtitle: run.date,
        minutes: secondsToMinutes(durationToSeconds(run.duration)),
        durationRaw: run.duration,
      })),
    [runsHistory]
  );

  const avgSec = durationToSeconds(stats.averageRunTime);
  const longSec = durationToSeconds(stats.longestRun);
  const gaugePercent = longSec > 0 ? Math.min(100, Math.round((avgSec / longSec) * 100)) : 0;

  const runStatus = useMemo(() => {
    if (processing.status === 'running') {
      return {
        label: 'Pipeline in progress',
        tone: 'progress',
        success: null,
      };
    }
    if (processing.status === 'completed') {
      return { label: 'Latest run: successful', tone: 'ok', success: true };
    }
    if (processing.status === 'stopped') {
      return { label: 'Latest run: halted', tone: 'bad', success: false };
    }
    return {
      label: stats.lastRunSuccessful ? 'Last run: successful' : 'Last run: failed',
      tone: stats.lastRunSuccessful ? 'ok' : 'bad',
      success: stats.lastRunSuccessful,
    };
  }, [processing.status, stats.lastRunSuccessful]);

  return (
    <div className="space-y-6">
      <p className="font-data text-sm text-slate-500">
        Command center metrics from{' '}
        <code className="text-cyan-700/90">dashboard.stats</code> and{' '}
        <code className="text-cyan-700/90">dashboard.runsHistory</code>.
      </p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total runs"
          icon={Radar}
          accent="cyan"
          className="sm:col-span-1"
        >
          <motion.p
            className="font-display text-4xl font-black tabular-nums text-cyan-300 [text-shadow:0_0_24px_rgba(34,211,238,0.45)] md:text-5xl"
            initial={{ opacity: 0.6, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
          >
            {totalRunsAnimated}
          </motion.p>
        </StatCard>

        <StatCard
          title="Longest run"
          icon={Trophy}
          accent="gold"
          className="sm:col-span-1"
        >
          <p className="font-data text-3xl font-bold tabular-nums text-amber-200 [text-shadow:0_0_20px_rgba(251,191,36,0.35)] md:text-4xl">
            {stats.longestRun}
          </p>
          <p className="font-data mt-2 text-[10px] uppercase tracking-wider text-amber-500/60">
            Gold / cyan highlight
          </p>
        </StatCard>

        <StatCard title="Average run time" icon={Timer} accent="blue" className="sm:col-span-1">
          <div className="flex flex-col items-center gap-3">
            <AverageGauge percent={gaugePercent} label={stats.averageRunTime} />
            <p className="font-data text-center text-[10px] text-slate-500">
              vs longest ({stats.longestRun}) · gauge = avg / longest
            </p>
          </div>
        </StatCard>

        <StatCard title="Current run status" icon={Activity} accent="slate" className="sm:col-span-1">
          <RunStatusBar status={runStatus} />
        </StatCard>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-4 backdrop-blur-md md:p-6">
        <div className="mb-4 flex items-center justify-between gap-2">
          <h3 className="font-display text-xs font-bold uppercase tracking-[0.2em] text-blue-400/90">
            Run duration history
          </h3>
          <span className="font-data text-[10px] text-slate-600">Neon trace · minutes</span>
        </div>
        <div className="h-[300px] w-full min-w-0">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
              <defs>
                <linearGradient id="missionNeonFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.45} />
                  <stop offset="55%" stopColor="#3b82f6" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="#1e3a8a" stopOpacity={0} />
                </linearGradient>
                <filter id="lineGlow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <CartesianGrid strokeDasharray="3 6" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={{ stroke: '#334155' }}
              />
              <YAxis
                tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={{ stroke: '#334155' }}
                tickFormatter={(v) => `${v}m`}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const row = payload[0].payload;
                  const sec = durationToSeconds(row.durationRaw);
                  return (
                    <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                      <p className="font-data text-xs font-semibold text-cyan-200">{row.label}</p>
                      <p className="font-data text-[10px] text-slate-500">{row.subtitle}</p>
                      <p className="font-data mt-1 text-sm text-slate-200">
                        Duration: <span className="text-blue-300">{row.durationRaw}</span> (
                        {formatSecondsAsHMS(sec)})
                      </p>
                    </div>
                  );
                }}
              />
              <Area
                type="monotone"
                dataKey="minutes"
                stroke="#38bdf8"
                strokeWidth={2.5}
                fill="url(#missionNeonFill)"
                filter="url(#lineGlow)"
                dot={{ r: 3, fill: '#7dd3fc', stroke: '#0ea5e9', strokeWidth: 1 }}
                activeDot={{ r: 5, fill: '#e0f2fe', stroke: '#38bdf8', strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, icon: Icon, children, accent, className = '' }) {
  const ring =
    accent === 'gold'
      ? 'shadow-[inset_0_0_0_1px_rgba(251,191,36,0.15)]'
      : accent === 'cyan'
        ? 'shadow-[inset_0_0_0_1px_rgba(34,211,238,0.12)]'
        : accent === 'blue'
          ? 'shadow-[inset_0_0_0_1px_rgba(59,130,246,0.12)]'
          : '';
  const iconColor =
    accent === 'gold'
      ? 'text-amber-400'
      : accent === 'cyan'
        ? 'text-cyan-400'
        : accent === 'blue'
          ? 'text-blue-400'
          : 'text-slate-400';

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/35 p-5 backdrop-blur-md ${ring} ${className}`}
    >
      <div className="pointer-events-none absolute -right-12 -top-12 h-32 w-32 rounded-full bg-blue-500/5 blur-2xl" />
      <div className="relative mb-3 flex items-center gap-2">
        <Icon className={`h-5 w-5 ${iconColor}`} strokeWidth={1.25} />
        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">
          {title}
        </h3>
      </div>
      <div className="relative">{children}</div>
    </div>
  );
}

function AverageGauge({ percent, label }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const offset = c - (percent / 100) * c;

  return (
    <div className="relative flex h-36 w-36 items-center justify-center">
      <svg className="h-36 w-36 -rotate-90" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={r} fill="none" stroke="#1e293b" strokeWidth="10" />
        <circle
          cx="60"
          cy="60"
          r={r}
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
          filter="url(#gaugeGlow)"
        />
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#3b82f6" />
          </linearGradient>
          <filter id="gaugeGlow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <p className="font-data text-lg font-bold tabular-nums text-cyan-100">{label}</p>
        <p className="font-data text-[10px] text-slate-500">{percent}% of longest</p>
      </div>
    </div>
  );
}

function RunStatusBar({ status }) {
  const { label, tone, success } = status;

  const barClass =
    tone === 'progress'
      ? 'from-cyan-600/40 to-blue-600/30'
      : tone === 'ok'
        ? 'from-emerald-600/50 to-cyan-600/30'
        : 'from-red-600/45 to-amber-600/25';

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {tone === 'progress' && <Loader2 className="h-5 w-5 animate-spin text-cyan-400" />}
        {success === true && <CheckCircle2 className="h-5 w-5 text-emerald-400" />}
        {success === false && <XCircle className="h-5 w-5 text-red-400" />}
        {success === null && tone !== 'progress' && (
          <Activity className="h-5 w-5 text-slate-500" />
        )}
        <p className="font-data text-sm font-medium text-slate-200">{label}</p>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-900 ring-1 ring-slate-800">
        <motion.div
          className={`h-full rounded-full bg-gradient-to-r ${barClass}`}
          initial={{ width: '0%' }}
          animate={{ width: tone === 'progress' ? '72%' : '100%' }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
        />
      </div>
      <p className="font-data text-[10px] text-slate-600">
        Live: <code className="text-slate-500">processing.status</code> · Snapshot:{' '}
        <code className="text-slate-500">dashboard.stats.lastRunSuccessful</code>
      </p>
    </div>
  );
}
