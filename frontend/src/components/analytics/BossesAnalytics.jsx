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
import { Swords, Skull, Timer, Target } from 'lucide-react';
import { useCountUp } from '../../hooks/useCountUp';
import { durationToSeconds, formatSecondsAsHMS } from '../../utils/duration';

const BAR_FILLS = ['#3b82f6', '#4f46e5', '#7c3aed', '#9333ea', '#a855f7', '#6366f1'];

function isDefeatedStatus(status) {
  return String(status ?? '').toLowerCase() === 'defeated';
}

/**
 * BOSSES — Tactical gallery + duration chart. Data: dashboard.bosses (name, lifespan, status only).
 */
export default function BossesAnalytics({ data }) {
  const bosses = data.dashboard.bosses ?? [];

  const enriched = useMemo(() => {
    return bosses.map((b, i) => {
      const sec = durationToSeconds(b.lifespan);
      const defeated = isDefeatedStatus(b.status);
      return {
        ...b,
        lifespanSec: sec,
        defeated,
        winRatePct: defeated ? 100 : 0,
        encounters: 1,
        barFill: BAR_FILLS[i % BAR_FILLS.length],
      };
    });
  }, [bosses]);

  const maxLifespanSec = useMemo(
    () => enriched.reduce((m, b) => Math.max(m, b.lifespanSec), 0),
    [enriched]
  );

  const totalEncounters = bosses.length;
  const defeatedCount = enriched.filter((b) => b.defeated).length;
  const globalKillRatePct =
    totalEncounters > 0 ? Math.round((defeatedCount / totalEncounters) * 100) : 0;

  const deadliest = useMemo(() => {
    if (!enriched.length) return { name: '—', detail: 'No boss data' };
    const minWin = Math.min(...enriched.map((b) => b.winRatePct));
    const pool = enriched.filter((b) => b.winRatePct === minWin);
    if (minWin === 100) {
      return { name: '—', detail: 'All targets defeated' };
    }
    const top = pool.reduce((a, b) => (b.lifespanSec > a.lifespanSec ? b : a));
    return { name: top.name, detail: 'Lowest clear rate (active)' };
  }, [enriched]);

  const chartData = useMemo(
    () =>
      enriched.map((b) => ({
        name: b.name,
        minutes: Math.round((b.lifespanSec / 60) * 100) / 100,
        seconds: b.lifespanSec,
        lifespanLabel: b.lifespan,
        fill: b.barFill,
      })),
    [enriched]
  );

  const totalAnimated = useCountUp(totalEncounters, 1200);
  const killRateAnimated = useCountUp(globalKillRatePct, 1400);

  return (
    <div className="space-y-8">
      <header>
        <div className="flex items-center gap-2">
          <Swords className="h-5 w-5 text-cyan-400" strokeWidth={1.25} aria-hidden />
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
            Boss analytics
          </p>
        </div>
        <h3 className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl">
          Threat assessment grid
        </h3>
        <p className="font-data mt-2 text-sm text-slate-500">
          Sourced from <code className="text-cyan-700/90">dashboard.bosses</code> · name, lifespan,
          status
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SummaryMetric
          icon={Target}
          title="Total boss encounters"
          subtitle="Entries in boss history"
          accent="cyan"
        >
          <p className="font-data text-3xl font-bold tabular-nums text-cyan-200 md:text-4xl">
            {totalAnimated}
          </p>
        </SummaryMetric>

        <SummaryMetric
          icon={Skull}
          title="Global kill rate"
          subtitle='Share with status "Defeated"'
          accent="blue"
        >
          <div className="flex items-baseline gap-1">
            <span className="font-data text-3xl font-bold tabular-nums text-slate-100 md:text-4xl">
              {killRateAnimated}
            </span>
            <span className="font-data text-xl font-semibold text-blue-400/80">%</span>
          </div>
        </SummaryMetric>

        <SummaryMetric icon={Timer} title="Deadliest boss" subtitle={deadliest.detail} accent="slate">
          <p className="font-display text-lg font-bold uppercase tracking-wide text-slate-100 md:text-xl">
            {deadliest.name}
          </p>
        </SummaryMetric>
      </div>

      <div>
        <h4 className="font-display mb-4 text-xs font-bold uppercase tracking-[0.2em] text-blue-400/90">
          Boss gallery
        </h4>
        {enriched.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-800 bg-slate-950/40 py-16 text-center">
            <p className="font-data text-sm text-slate-500">No bosses in catalog.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {enriched.map((boss) => (
              <BossProfileCard key={boss.id ?? boss.name} boss={boss} maxLifespanSec={maxLifespanSec} />
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-slate-800 bg-transparent p-4 backdrop-blur-md md:p-6">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
          <div className="flex items-center gap-2">
            <Timer className="h-4 w-4 text-blue-400" strokeWidth={1.25} aria-hidden />
            <h4 className="font-display text-xs font-bold uppercase tracking-[0.2em] text-blue-400/90">
              Avg. fight duration per boss
            </h4>
          </div>
          <span className="font-data text-[10px] text-slate-600">Lifespan · minutes</span>
        </div>
        <div className="h-[320px] w-full min-w-0">
          {chartData.length === 0 ? (
            <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">
              <p className="font-data text-sm text-slate-500">No chart data.</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 48 }}>
                <CartesianGrid strokeDasharray="3 6" stroke="#334155" strokeOpacity={0.6} vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                  axisLine={{ stroke: '#475569' }}
                  tickLine={{ stroke: '#475569' }}
                  angle={-28}
                  textAnchor="end"
                  height={56}
                  interval={0}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                  axisLine={{ stroke: '#475569' }}
                  tickLine={{ stroke: '#475569' }}
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
                  cursor={{ fill: 'rgba(51, 65, 85, 0.25)' }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const row = payload[0].payload;
                    return (
                      <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-md">
                        <p className="font-data text-xs font-semibold text-cyan-200">{row.name}</p>
                        <p className="font-data mt-1 tabular-nums text-sm text-slate-200">
                          {row.lifespanLabel} · {formatSecondsAsHMS(row.seconds)}
                        </p>
                        <p className="font-data text-[10px] text-slate-500">{row.minutes} min</p>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="minutes" name="Duration" radius={[6, 6, 0, 0]} maxBarSize={56}>
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

function SummaryMetric({ icon: Icon, title, subtitle, accent, children }) {
  const ring = {
    cyan: 'ring-cyan-500/20',
    blue: 'ring-blue-500/20',
    slate: 'ring-slate-600/30',
  };
  const iconCls = {
    cyan: 'text-cyan-400',
    blue: 'text-blue-400',
    slate: 'text-slate-400',
  };

  return (
    <div
      className={`rounded-2xl border border-slate-800 bg-slate-900/25 p-5 backdrop-blur-md ring-1 ${ring[accent]}`}
    >
      <div className="mb-3 flex items-center gap-2">
        <Icon className={`h-5 w-5 ${iconCls[accent]}`} strokeWidth={1.25} />
        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">
          {title}
        </h3>
      </div>
      <p className="font-data mb-3 text-[10px] text-slate-600">{subtitle}</p>
      <div className="font-data">{children}</div>
    </div>
  );
}

function BossProfileCard({ boss, maxLifespanSec }) {
  const defeated = boss.defeated;
  const badgeLabel = defeated ? 'DEFEATED' : 'ACTIVE';
  const badgeClass = defeated
    ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-300 shadow-[0_0_16px_rgba(52,211,153,0.45)]'
    : 'border-red-500/50 bg-red-500/15 text-red-300 shadow-[0_0_16px_rgba(248,113,113,0.4)]';

  const gaugePct = maxLifespanSec > 0 ? Math.round((boss.lifespanSec / maxLifespanSec) * 100) : 0;

  return (
    <motion.article
      initial={false}
      whileHover={{ scale: 1.03 }}
      transition={{ type: 'spring', stiffness: 400, damping: 24 }}
      className="group relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/30 p-5 shadow-[0_0_0_1px_rgba(15,23,42,0.8)] backdrop-blur-md ring-1 ring-slate-700/40 hover:border-cyan-500/25 hover:shadow-[0_0_28px_rgba(59,130,246,0.22),0_0_48px_rgba(139,92,246,0.12)]"
    >
      <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
        <div className="absolute inset-0 bg-linear-to-br from-blue-500/5 via-transparent to-purple-500/8" />
      </div>

      <div className="relative flex flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h3 className="font-display text-lg font-extrabold uppercase tracking-[0.12em] text-slate-100 md:text-xl">
            {boss.name}
          </h3>
          <span
            className={`shrink-0 rounded-md border px-2.5 py-1 font-display text-[9px] font-bold tracking-[0.25em] ${badgeClass}`}
          >
            {badgeLabel}
          </span>
        </div>

        <div>
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="font-data text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Avg. lifespan
            </span>
            <span className="font-data text-xs tabular-nums text-cyan-200/90">{boss.lifespan}</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-800/90">
            <motion.div
              className="h-full rounded-full bg-linear-to-r from-blue-500 via-indigo-500 to-purple-500 shadow-[0_0_12px_rgba(99,102,241,0.5)]"
              initial={{ width: 0 }}
              animate={{ width: `${gaugePct}%` }}
              transition={{ duration: 0.9, ease: 'easeOut' }}
            />
          </div>
        </div>

        <div className="flex gap-6 border-t border-slate-800/80 pt-3">
          <div>
            <p className="font-data text-[9px] uppercase tracking-wider text-slate-600">Total encounters</p>
            <p className="font-data mt-0.5 text-sm font-semibold tabular-nums text-slate-200">
              {boss.encounters}
            </p>
          </div>
          <div>
            <p className="font-data text-[9px] uppercase tracking-wider text-slate-600">Win rate</p>
            <p className="font-data mt-0.5 text-sm font-semibold tabular-nums text-slate-200">
              {boss.winRatePct}%
            </p>
          </div>
        </div>
      </div>
    </motion.article>
  );
}
