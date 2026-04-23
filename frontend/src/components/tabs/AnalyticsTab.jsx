import { motion } from 'framer-motion';
import GeneralMissionStats from '../analytics/GeneralMissionStats';
import BossesAnalytics from '../analytics/BossesAnalytics';
import ItemsPowerLab from '../analytics/ItemsPowerLab';

const SUB_TABS = [
  { id: 'general', label: 'GENERAL' },
  { id: 'bosses', label: 'BOSSES' },
  { id: 'items', label: 'ITEMS' },
];

/**
 * ANALYTICS — secondary nav + sub-views. Writes ui.analyticsSubTab.
 * GENERAL: dashboard.runsHistory, dashboard.bosses, dashboard.items.
 * BOSSES: dashboard.bosses + runsHistory + dashboard.items (gear correlations).
 * ITEMS: dashboard.items + runsHistory — run duration / survival simulator.
 */
export default function AnalyticsTab({ data, onPatch }) {
  const sub = data.ui.analyticsSubTab;

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className="mx-auto max-w-[1800px] px-4 py-8 md:py-10"
    >
      <header className="mb-6">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Intelligence
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Analytics deck
        </h2>

      </header>

      <nav
        className="mb-8 flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-950/40 p-2 backdrop-blur-md"
        aria-label="Analytics sections"
      >
        {SUB_TABS.map((t) => {
          const active = sub === t.id;
          const itemsTab = t.id === 'items';
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={`relative flex-1 rounded-lg px-4 py-2.5 font-display text-[10px] font-bold tracking-[0.2em] transition sm:flex-none sm:min-w-[100px] ${
                active
                  ? itemsTab
                    ? 'bg-violet-500/15 text-violet-200 ring-1 ring-violet-500/40'
                    : 'bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/35'
                  : 'text-slate-500 hover:bg-slate-900/60 hover:text-slate-300'
              }`}
              onClick={() =>
                onPatch({
                  ui: { ...data.ui, analyticsSubTab: t.id },
                })
              }
            >
              {t.label}
            </button>
          );
        })}
      </nav>

      <motion.div
        key={sub}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22 }}
      >
        {sub === 'general' && <GeneralMissionStats data={data} />}

        {sub === 'bosses' && <BossesAnalytics data={data} />}

        {sub === 'items' && <ItemsPowerLab data={data} />}
      </motion.div>
    </motion.div>
  );
}
