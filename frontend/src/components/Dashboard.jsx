import { BarChart3, Swords, Package, Clock, Trophy, Hash } from 'lucide-react';

const TABS = [
  { id: 'summary', label: 'Summary', icon: BarChart3 },
  { id: 'combat', label: 'Combat', icon: Swords },
  { id: 'inventory', label: 'Inventory', icon: Package },
];

/**
 * Post-processing dashboard — reads dashboard.* and ui.dashboardActiveTab.
 * BACKEND: Replace mock numbers with API payload mapped into dashboard.* shape.
 */
export default function Dashboard({ data, onPatch }) {
  const { dashboard, ui } = data;
  const { stats, items, bosses, runsHistory } = dashboard;

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-100">Mission Dashboard</h2>
          <p className="text-sm text-slate-500">
            Populated when processing completes — BACKEND: refresh this block from your report API.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6 flex flex-wrap gap-2 border-b border-slate-800 pb-1">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`flex items-center gap-2 rounded-t-xl px-4 py-2.5 text-sm font-semibold transition ${
              ui.dashboardActiveTab === id
                ? 'border border-b-0 border-slate-800 bg-slate-800/80 text-emerald-400'
                : 'text-slate-500 hover:bg-slate-800/50 hover:text-slate-300'
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
            hint="BACKEND: dashboard.stats.totalRuns"
          />
          <StatCard
            icon={Clock}
            label="Avg. run time"
            value={stats.averageRunTime}
            hint="BACKEND: dashboard.stats.averageRunTime"
          />
          <StatCard
            icon={Trophy}
            label="Longest run"
            value={stats.longestRun}
            hint="BACKEND: dashboard.stats.longestRun"
          />
          <StatCard
            icon={Package}
            label="Items found"
            value={stats.totalItemsFound}
            hint="BACKEND: dashboard.stats.totalItemsFound"
          />
          <div className="sm:col-span-2 lg:col-span-4 rounded-2xl border border-slate-800 bg-slate-800/30 p-4">
            <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-500">
              Run history
            </h3>
            <ul className="divide-y divide-slate-800">
              {runsHistory.map((run) => (
                <li key={run.id} className="flex flex-wrap justify-between gap-2 py-3 text-sm">
                  <span className="font-mono text-slate-300">{run.id}</span>
                  <span className="text-slate-500">{run.date}</span>
                  <span className="text-emerald-400/90">{run.duration}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {ui.dashboardActiveTab === 'combat' && (
        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-800/20">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-900/80 text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-3 font-bold">Boss name</th>
                <th className="px-4 py-3 font-bold">Lifespan</th>
                <th className="px-4 py-3 font-bold">Status</th>
              </tr>
            </thead>
            <tbody>
              {bosses.map((b) => (
                <tr key={b.id} className="border-b border-slate-800/80 last:border-0 hover:bg-slate-800/40">
                  <td className="px-4 py-3 font-medium text-slate-200">{b.name}</td>
                  <td className="px-4 py-3 font-mono text-slate-400">{b.lifespan}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                        b.status === 'Alive'
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-emerald-500/20 text-emerald-400'
                      }`}
                    >
                      {b.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-slate-800 px-4 py-2 text-[10px] text-slate-600">
            BACKEND: map boss rows to dashboard.bosses[]
          </p>
        </div>
      )}

      {ui.dashboardActiveTab === 'inventory' && (
        <div className="space-y-4">
          <p className="text-xs text-slate-500">
            Popularity bars — BACKEND: dashboard.items[].popularity (0–100) and .impact
          </p>
          <ul className="space-y-3">
            {items.map((item) => (
              <li
                key={item.id}
                className="rounded-2xl border border-slate-800 bg-slate-800/30 p-4"
              >
                <div className="mb-2 flex justify-between gap-2 text-sm">
                  <span className="font-semibold text-slate-200">{item.name}</span>
                  <span className="text-slate-500">
                    Impact: <span className="text-slate-300">{item.impact}</span>
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-950">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-600 to-emerald-500 transition-all duration-500"
                    style={{ width: `${item.popularity}%` }}
                  />
                </div>
                <p className="mt-1 text-right text-xs text-slate-500">{item.popularity}%</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon: Icon, label, value, hint }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-800/40 p-5 transition hover:border-slate-700">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-500">{label}</span>
        <Icon className="h-4 w-4 text-blue-400/80" />
      </div>
      <p className="text-2xl font-bold text-slate-100">{value}</p>
      <p className="mt-2 text-[10px] leading-tight text-slate-600">{hint}</p>
    </div>
  );
}
