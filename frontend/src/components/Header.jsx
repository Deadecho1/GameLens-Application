import { Settings, ChevronDown, Crosshair } from 'lucide-react';

/**
 * Header — mission selection + engine entry. Config (add game/version) via Gear → ConfigSidebar.
 * Reads/writes setup.* and ui.* through onPatch.
 */
export default function Header({ data, onPatch }) {
  const { setup, ui } = data;

  return (
    <header className="sticky top-0 z-40 border-b border-blue-500/10 bg-slate-950/70 px-4 py-3 backdrop-blur-xl">
      <div className="relative mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4">
        <div className="pointer-events-none absolute inset-x-0 top-full h-px bg-gradient-to-r from-transparent via-cyan-500/25 to-transparent" />

        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="relative flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-500/20 bg-slate-900/90 text-cyan-400 shadow-[0_0_24px_rgba(34,211,238,0.12)]">
            <Crosshair className="h-5 w-5" aria-hidden />
          </div>
          <div>
            <h1 className="font-display text-lg font-extrabold tracking-tight text-slate-100">
              GAMELENS
            </h1>
            <p className="font-data text-[10px] font-medium uppercase tracking-[0.25em] text-blue-500/70">
              Engine console
            </p>
          </div>
        </div>

        {/* Mission selection */}
        <div className="flex flex-1 flex-wrap items-center justify-center gap-3 md:gap-5">
          <div className="relative min-w-[200px]">
            <label
              htmlFor="game-select"
              className="font-display mb-1.5 block text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80"
            >
              Mission select
            </label>
            <div className="relative">
              <select
                id="game-select"
                className="font-data w-full cursor-pointer appearance-none rounded-xl border border-slate-800 bg-slate-900/90 py-2.5 pl-3 pr-10 text-sm text-cyan-100 outline-none ring-cyan-400/20 transition focus:border-blue-500/50 focus:ring-2"
                value={setup.selectedGame}
                onChange={(e) =>
                  onPatch({
                    setup: { ...setup, selectedGame: e.target.value },
                  })
                }
              >
                {setup.games.map((g) => (
                  <option key={g} value={g} className="bg-slate-900">
                    {g}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-500/60" />
            </div>
          </div>

          <div className="relative hidden h-8 w-px bg-slate-800 sm:block" aria-hidden />

          <div className="relative min-w-[220px]">
            <label
              htmlFor="version-select"
              className="font-display mb-1.5 block text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80"
            >
              Build target
            </label>
            <div className="relative">
              <select
                id="version-select"
                className="font-data w-full cursor-pointer appearance-none rounded-xl border border-slate-800 bg-slate-900/90 py-2.5 pl-3 pr-10 text-sm text-cyan-100 outline-none ring-cyan-400/20 transition focus:border-blue-500/50 focus:ring-2"
                value={setup.selectedVersion}
                onChange={(e) =>
                  onPatch({
                    setup: { ...setup, selectedVersion: e.target.value },
                  })
                }
              >
                {setup.versions.map((v) => (
                  <option key={v} value={v} className="bg-slate-900">
                    {v}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-500/60" />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            title="Mission catalog & registration"
            className="flex h-11 w-11 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/80 text-slate-400 transition hover:border-cyan-500/40 hover:text-cyan-400"
            onClick={() =>
              onPatch({
                ui: { ...ui, configSidebarOpen: !ui.configSidebarOpen },
              })
            }
            aria-expanded={ui.configSidebarOpen}
            aria-label="Open configuration"
          >
            <Settings className="h-5 w-5" />
          </button>

          <div className="gl-radial-glow relative inline-block rounded-2xl p-[1px]">
            <button
              type="button"
              className="relative z-[1] rounded-2xl border border-blue-500/50 bg-gradient-to-b from-blue-600 to-blue-700 px-6 py-2.5 font-display text-xs font-bold uppercase tracking-[0.15em] text-white shadow-[0_0_32px_rgba(59,130,246,0.35)] transition hover:brightness-110 active:scale-[0.98]"
              onClick={() =>
                onPatch({
                  ui: { ...ui, processingModalOpen: true },
                })
              }
            >
              Process clip
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
