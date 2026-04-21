import { motion } from 'framer-motion';
import { Swords, Package, History, Clock, Trophy, Hash } from 'lucide-react';

/**
 * ANALYTICS — bento grid from dashboard.* (single view, no sub-tabs).
 * BACKEND: hydrate all cells from report API.
 */
export default function AnalyticsTab({ data }) {
  const { stats, bosses, items, runsHistory } = data.dashboard;

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className="mx-auto max-w-[1800px] px-4 py-8 md:py-10"
    >
      <header className="mb-8">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Intelligence
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Analytics deck
        </h2>
        <p className="font-data mt-2 text-sm text-slate-500">
          Live mirror of <code className="text-cyan-600/75">dashboard.*</code>
        </p>
      </header>

      <div className="grid auto-rows-fr grid-cols-1 gap-4 md:grid-cols-4 md:gap-5 lg:grid-cols-6">
        {/* Headline metrics — wide bento */}
        <BentoCell
          className="md:col-span-4 lg:col-span-3"
          title="Run throughput"
          icon={Hash}
        >
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Metric label="Total runs" value={stats.totalRuns} />
            <Metric label="Avg time" value={stats.averageRunTime} />
            <Metric label="Longest" value={stats.longestRun} />
            <Metric label="Items" value={stats.totalItemsFound} />
          </div>
        </BentoCell>

        <BentoCell className="md:col-span-2 lg:col-span-3" title="Boss matrix" icon={Swords}>
          <ul className="space-y-2 font-data text-sm">
            {bosses.map((b) => (
              <li
                key={b.id}
                className="flex items-center justify-between gap-2 rounded-xl border border-slate-800/80 bg-black/25 px-3 py-2"
              >
                <span className="font-semibold text-slate-200">{b.name}</span>
                <span className="text-cyan-500/70">{b.lifespan}</span>
                <span
                  className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${
                    b.status === 'Alive'
                      ? 'bg-amber-500/15 text-amber-400'
                      : 'bg-emerald-500/15 text-emerald-400'
                  }`}
                >
                  {b.status}
                </span>
              </li>
            ))}
          </ul>
        </BentoCell>

        {/* Items — tall */}
        <BentoCell className="md:col-span-2 lg:col-span-2" title="Item telemetry" icon={Package}>
          <ul className="space-y-3">
            {items.map((item) => (
              <li key={item.id} className="rounded-xl border border-slate-800/60 bg-black/20 p-3">
                <div className="mb-2 flex justify-between font-data text-xs">
                  <span className="text-slate-200">{item.name}</span>
                  <span className="text-slate-500">{item.impact}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-950 ring-1 ring-slate-800">
                  <motion.div
                    className="gl-neon-bar-fill h-full rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${item.popularity}%` }}
                    transition={{ duration: 0.7, ease: 'easeOut' }}
                  />
                </div>
                <p className="mt-1 text-right font-data text-[10px] text-slate-500">{item.popularity}%</p>
              </li>
            ))}
          </ul>
        </BentoCell>

        {/* Run history — wide */}
        <BentoCell className="md:col-span-2 lg:col-span-4" title="Run history" icon={History}>
          <ul className="divide-y divide-slate-800/80 font-data text-sm">
            {runsHistory.map((run) => (
              <li key={run.id} className="flex flex-wrap items-center justify-between gap-2 py-3 first:pt-0">
                <span className="text-cyan-200/90">{run.id}</span>
                <span className="flex items-center gap-1 text-slate-500">
                  <Clock className="h-3.5 w-3.5" />
                  {run.date}
                </span>
                <span className="text-blue-400/90">{run.duration}</span>
              </li>
            ))}
          </ul>
        </BentoCell>

        {/* Accent tile */}
        <BentoCell className="md:col-span-2 lg:col-span-2" title="High score" icon={Trophy}>
          <p className="font-display text-3xl font-bold text-cyan-300/90">{stats.longestRun}</p>
          <p className="font-data mt-2 text-xs text-slate-500">Longest recorded sortie · dashboard.stats.longestRun</p>
        </BentoCell>
      </div>
    </motion.div>
  );
}

function BentoCell({ title, icon: Icon, children, className = '' }) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-slate-800/90 bg-slate-900/40 p-5 shadow-[inset_0_1px_0_rgba(34,211,238,0.05)] backdrop-blur-md transition hover:border-cyan-500/20 ${className}`}
    >
      <div className="pointer-events-none absolute -right-20 -top-20 h-40 w-40 rounded-full bg-blue-600/10 blur-3xl" />
      <div className="relative mb-4 flex items-center gap-2">
        <Icon className="h-4 w-4 text-cyan-500/70" />
        <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/85">
          {title}
        </h3>
      </div>
      <div className="relative">{children}</div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-black/30 px-3 py-3">
      <p className="font-data text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="font-data mt-1 text-lg font-bold text-slate-100">{value}</p>
    </div>
  );
}
