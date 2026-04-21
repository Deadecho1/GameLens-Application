import { motion } from 'framer-motion';
import { BarChart3, Swords, Package, Clock, Trophy, Hash } from 'lucide-react';

const TABS = [
  { id: 'summary', label: 'Summary', icon: BarChart3 },
  { id: 'combat', label: 'Combat', icon: Swords },
  { id: 'inventory', label: 'Inventory', icon: Package },
];

/**
 * Tactical dashboard — reads dashboard.* / ui.dashboardActiveTab.
 * BACKEND: hydrate from report API when job completes.
 */
export default function Dashboard({ data, onPatch }) {
  const { dashboard, ui } = data;
  const { stats, items, bosses, runsHistory } = dashboard;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1600px] px-4 py-10"
    >
      <div className="relative mb-8 overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/60 p-6 backdrop-blur-sm">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_80%_at_0%_0%,rgba(59,130,246,0.08),transparent)]" />
        <div className="relative">
          <h2 className="font-display text-2xl font-bold tracking-tight text-slate-100 md:text-3xl">
            Tactical dashboard
          </h2>
          <p className="font-data mt-2 max-w-2xl text-sm text-slate-500">
            Live intel from the last successful pipeline — BACKEND: map GET /report into{' '}
            <code className="text-cyan-600/70">dashboard.*</code>
          </p>
        </div>
      </div>

      <div className="mb-6 flex flex-wrap gap-2 border-b border-slate-800/80 pb-1">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`font-display flex items-center gap-2 rounded-t-xl px-4 py-2.5 text-xs font-bold uppercase tracking-wider transition ${
              ui.dashboardActiveTab === id
                ? 'border border-b-0 border-cyan-500/30 bg-slate-900/90 text-cyan-400 shadow-[0_-4px_24px_rgba(34,211,238,0.08)]'
                : 'text-slate-500 hover:bg-slate-900/50 hover:text-slate-300'
            }`}
            onClick={() =>
              onPatch({
                ui: { ...ui, dashboardActiveTab: id },
              })
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {ui.dashboardActiveTab === 'summary' && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            icon={Hash}
            label="Total runs"
            value={stats.totalRuns}
            hint="dashboard.stats.totalRuns"
          />
          <StatCard
            icon={Clock}
            label="Avg. run time"
            value={stats.averageRunTime}
            hint="dashboard.stats.averageRunTime"
          />
          <StatCard
            icon={Trophy}
            label="Longest run"
            value={stats.longestRun}
            hint="dashboard.stats.longestRun"
          />
          <StatCard
            icon={Package}
            label="Items found"
            value={stats.totalItemsFound}
            hint="dashboard.stats.totalItemsFound"
          />
          <div className="sm:col-span-2 lg:col-span-4 rounded-2xl border border-slate-800 bg-slate-950/50 p-5 backdrop-blur-sm">
            <h3 className="font-display mb-4 text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
              Run history
            </h3>
            <ul className="divide-y divide-slate-800/80">
              {runsHistory.map((run) => (
                <li
                  key={run.id}
                  className="flex flex-wrap justify-between gap-2 py-3 font-data text-sm"
                >
                  <span className="text-cyan-200/90">{run.id}</span>
                  <span className="text-slate-500">{run.date}</span>
                  <span className="text-blue-400/90">{run.duration}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {ui.dashboardActiveTab === 'combat' && (
        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/40 backdrop-blur-sm">
          <table className="w-full text-left font-data text-sm">
            <thead className="border-b border-slate-800 bg-black/30 text-[10px] uppercase tracking-wider text-blue-500/80">
              <tr>
                <th className="px-4 py-3 font-bold">Boss</th>
                <th className="px-4 py-3 font-bold">Lifespan</th>
                <th className="px-4 py-3 font-bold">Status</th>
              </tr>
            </thead>
            <tbody>
              {bosses.map((b) => (
                <tr
                  key={b.id}
                  className="border-b border-slate-800/60 last:border-0 hover:bg-slate-900/40"
                >
                  <td className="px-4 py-3 font-medium text-slate-200">{b.name}</td>
                  <td className="px-4 py-3 text-cyan-500/70">{b.lifespan}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                        b.status === 'Alive'
                          ? 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20'
                          : 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20'
                      }`}
                    >
                      {b.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-slate-800 px-4 py-2 font-data text-[10px] text-slate-600">
            Rows: <code className="text-slate-500">dashboard.bosses[]</code>
          </p>
        </div>
      )}

      {ui.dashboardActiveTab === 'inventory' && (
        <div className="space-y-4">
          <p className="font-data text-xs text-slate-500">
            Popularity telemetry — <code className="text-cyan-700/80">dashboard.items[]</code>
          </p>
          <ul className="space-y-3">
            {items.map((item) => (
              <li
                key={item.id}
                className="rounded-2xl border border-slate-800 bg-slate-950/50 p-4 backdrop-blur-sm transition hover:border-cyan-500/20"
              >
                <div className="mb-2 flex justify-between gap-2 font-data text-sm">
                  <span className="font-semibold text-slate-200">{item.name}</span>
                  <span className="text-slate-500">
                    Impact{' '}
                    <span className="text-cyan-400/80">{item.impact}</span>
                  </span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-black/60 ring-1 ring-slate-800">
                  <motion.div
                    className="gl-neon-bar-fill h-full rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${item.popularity}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                  />
                </div>
                <p className="mt-1 text-right font-data text-xs text-slate-500">{item.popularity}%</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
}

function StatCard({ icon: Icon, label, value, hint }) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/50 p-5 backdrop-blur-sm transition hover:border-blue-500/25">
      <div className="pointer-events-none absolute -right-8 -top-8 h-24 w-24 rounded-full bg-blue-500/10 blur-2xl transition group-hover:bg-cyan-500/10" />
      <div className="relative mb-3 flex items-center justify-between">
        <span className="font-display text-[10px] font-bold uppercase tracking-[0.15em] text-blue-500/80">
          {label}
        </span>
        <Icon className="h-4 w-4 text-cyan-500/60" />
      </div>
      <p className="font-data relative text-2xl font-bold text-slate-100">{value}</p>
      <p className="font-data relative mt-2 text-[10px] leading-tight text-slate-600">{hint}</p>
    </div>
  );
}
