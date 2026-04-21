import { motion } from 'framer-motion';

const TABS = [
  { id: 'setup', label: 'SETUP' },
  { id: 'process', label: 'PROCESS' },
  { id: 'analytics', label: 'ANALYTICS' },
];

/**
 * Console-style primary nav. Writes ui.activeMainTab only.
 */
export default function MainTabNav({ data, onPatch }) {
  const active = data.ui.activeMainTab;

  return (
    <nav
      className="relative z-20 border-b border-slate-800/80 bg-slate-950/55 px-3 py-3 backdrop-blur-xl"
      aria-label="Main console"
    >
      <div className="mx-auto flex max-w-[1800px] flex-wrap gap-2 md:gap-3">
        {TABS.map((tab) => {
          const isActive = active === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              className={`relative min-w-[120px] flex-1 rounded-xl border px-4 py-3 font-display text-xs font-bold tracking-[0.2em] transition sm:flex-none sm:min-w-[140px] md:px-8 ${
                isActive
                  ? 'border-cyan-500/40 bg-slate-900/80 text-cyan-300 shadow-[0_0_24px_rgba(34,211,238,0.12),inset_0_1px_0_rgba(34,211,238,0.12)]'
                  : 'border-slate-800/90 bg-slate-900/35 text-slate-500 hover:border-slate-700 hover:bg-slate-900/55 hover:text-slate-300'
              }`}
              onClick={() =>
                onPatch({
                  ui: { ...data.ui, activeMainTab: tab.id },
                })
              }
            >
              {isActive && (
                <motion.span
                  layoutId="main-tab-glow"
                  className="pointer-events-none absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-500 shadow-[0_0_12px_rgba(34,211,238,0.8)]"
                  transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                />
              )}
              <span className="relative z-[1]">{tab.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
