import { Plus, Sparkles } from 'lucide-react';

/**
 * Header — reads setup.* and ui.*; writes via onPatch.
 * BACKEND: Wire dropdowns to the same setup fields after API fetch.
 */
export default function Header({ data, onPatch }) {
  const { setup, ui } = data;

  return (
    <header
      className="sticky top-0 z-40 border-b border-slate-800 bg-slate-900/75 px-4 py-3 backdrop-blur-md transition-colors"
      style={{ boxShadow: '0 4px 30px rgba(0,0,0,0.35)' }}
    >
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-800 bg-slate-800/80 text-emerald-400 shadow-inner">
            <Sparkles className="h-5 w-5" aria-hidden />
          </div>
          <div>
            <h1 className="bg-gradient-to-r from-slate-100 to-slate-400 bg-clip-text font-black tracking-tight text-transparent">
              GameLens
            </h1>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
              Analysis Console
            </p>
          </div>
        </div>

        {/* Center: selectors */}
        <div className="flex flex-1 flex-wrap items-center justify-center gap-4 md:gap-6">
          <div className="flex items-center gap-2">
            <label htmlFor="game-select" className="text-xs font-bold uppercase text-slate-500">
              Game
            </label>
            <select
              id="game-select"
              className="max-w-[200px] rounded-xl border border-slate-800 bg-slate-800/90 px-3 py-2 text-sm text-slate-100 outline-none ring-emerald-500/30 transition focus:ring-2"
              value={setup.selectedGame}
              onChange={(e) =>
                onPatch({
                  setup: { ...setup, selectedGame: e.target.value },
                })
              }
            >
              {setup.games.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
            <button
              type="button"
              title="Add new game"
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-800 bg-slate-800 text-slate-300 transition hover:border-emerald-500/50 hover:text-emerald-400"
              onClick={() =>
                onPatch({
                  ui: { ...ui, addGameModalOpen: true, newGameNameDraft: '' },
                })
              }
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="version-select" className="text-xs font-bold uppercase text-slate-500">
              Version
            </label>
            <select
              id="version-select"
              className="max-w-[220px] rounded-xl border border-slate-800 bg-slate-800/90 px-3 py-2 text-sm text-slate-100 outline-none ring-blue-500/30 transition focus:ring-2"
              value={setup.selectedVersion}
              onChange={(e) =>
                onPatch({
                  setup: { ...setup, selectedVersion: e.target.value },
                })
              }
            >
              {setup.versions.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
            <button
              type="button"
              title="Add new version"
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-800 bg-slate-800 text-slate-300 transition hover:border-blue-500/50 hover:text-blue-400"
              onClick={() =>
                onPatch({
                  ui: { ...ui, addVersionModalOpen: true, newVersionNameDraft: '' },
                })
              }
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Process Clip */}
        <button
          type="button"
          className="rounded-2xl bg-gradient-to-r from-blue-600 to-emerald-600 px-6 py-2.5 text-sm font-bold uppercase tracking-wider text-white shadow-[0_0_24px_rgba(59,130,246,0.35)] transition hover:brightness-110 active:scale-[0.98]"
          onClick={() =>
            onPatch({
              ui: { ...ui, processingModalOpen: true },
            })
          }
        >
          Process Clip
        </button>
      </div>
    </header>
  );
}
