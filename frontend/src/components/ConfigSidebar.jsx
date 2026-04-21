import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, Database } from 'lucide-react';

/**
 * Mission catalog drawer — opens from Gear in header.
 * Writes ui.configSidebarOpen; triggers add-game / add-version modals (still in dataStore.ui.*).
 * BACKEND: Replace list sections with API-driven mission/build registry.
 */
export default function ConfigSidebar({ data, onPatch, onAddGame, onAddVersion }) {
  const { setup, ui } = data;
  const open = ui.configSidebarOpen;

  return (
    <AnimatePresence mode="sync">
      {open && (
        <motion.button
          key="config-backdrop"
          type="button"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[55] bg-slate-950/70 backdrop-blur-md"
          aria-label="Close config"
          onClick={() => onPatch({ ui: { ...ui, configSidebarOpen: false } })}
        />
      )}
      {open && (
        <motion.aside
          key="config-drawer"
          role="dialog"
          aria-modal="true"
          aria-label="Mission configuration"
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 28, stiffness: 320 }}
          className="fixed right-0 top-0 z-[56] flex h-full w-full max-w-md flex-col border-l border-cyan-500/15 bg-slate-950/85 shadow-[-12px_0_48px_rgba(0,0,0,0.5)] backdrop-blur-xl"
        >
            <div className="flex items-center justify-between border-b border-slate-800/80 px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-blue-500/30 bg-blue-500/10 text-cyan-400">
                  <Database className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="font-display text-sm font-bold uppercase tracking-widest text-slate-200">
                    Config
                  </h2>
                  <p className="font-data text-[10px] uppercase tracking-[0.2em] text-slate-500">
                    Mission registry
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-800 hover:text-cyan-400"
                onClick={() => onPatch({ ui: { ...ui, configSidebarOpen: false } })}
                aria-label="Close"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-6">
              <section className="mb-8">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-display text-xs font-bold uppercase tracking-[0.15em] text-blue-400/90">
                    Missions (games)
                  </h3>
                  <button
                    type="button"
                    className="font-data inline-flex items-center gap-1 rounded-lg border border-cyan-500/25 bg-cyan-500/5 px-2.5 py-1 text-[11px] font-semibold text-cyan-400 transition hover:bg-cyan-500/15"
                    onClick={() => {
                      onAddGame();
                      onPatch({ ui: { ...ui, configSidebarOpen: false } });
                    }}
                  >
                    <Plus className="h-3.5 w-3.5" />
                    New
                  </button>
                </div>
                <ul className="space-y-1 rounded-xl border border-slate-800/80 bg-slate-900/50 p-2 font-data text-sm">
                  {setup.games.map((g) => (
                    <li key={g}>
                      <button
                        type="button"
                        className={`w-full rounded-lg px-3 py-2.5 text-left transition ${
                          setup.selectedGame === g
                            ? 'bg-blue-500/15 text-cyan-300 ring-1 ring-blue-500/30'
                            : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                        }`}
                        onClick={() =>
                          onPatch({
                            setup: { ...setup, selectedGame: g },
                          })
                        }
                      >
                        {g}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-display text-xs font-bold uppercase tracking-[0.15em] text-blue-400/90">
                    Builds (versions)
                  </h3>
                  <button
                    type="button"
                    className="font-data inline-flex items-center gap-1 rounded-lg border border-cyan-500/25 bg-cyan-500/5 px-2.5 py-1 text-[11px] font-semibold text-cyan-400 transition hover:bg-cyan-500/15"
                    onClick={() => {
                      onAddVersion();
                      onPatch({ ui: { ...ui, configSidebarOpen: false } });
                    }}
                  >
                    <Plus className="h-3.5 w-3.5" />
                    New
                  </button>
                </div>
                <ul className="space-y-1 rounded-xl border border-slate-800/80 bg-slate-900/50 p-2 font-data text-sm">
                  {setup.versions.map((v) => (
                    <li key={v}>
                      <button
                        type="button"
                        className={`w-full rounded-lg px-3 py-2.5 text-left transition ${
                          setup.selectedVersion === v
                            ? 'bg-blue-500/15 text-cyan-300 ring-1 ring-blue-500/30'
                            : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                        }`}
                        onClick={() =>
                          onPatch({
                            setup: { ...setup, selectedVersion: v },
                          })
                        }
                      >
                        {v}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            </div>

            <p className="border-t border-slate-800/80 px-5 py-3 font-data text-[10px] leading-relaxed text-slate-600">
              BACKEND: sync lists from your catalog; &quot;New&quot; opens the register modal — POST then
              refresh <code className="text-slate-500">setup.games</code> /{' '}
              <code className="text-slate-500">setup.versions</code>.
            </p>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
