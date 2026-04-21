import { Activity, Crosshair } from 'lucide-react';

/**
 * Minimal shell header — logo + live status. All catalog UX lives under SETUP tab.
 * Status derives from processing.status for a believable “ops” readout.
 */
export default function Header({ data }) {
  const status = data.processing.status;
  const online =
    status === 'running' ? 'busy' : status === 'stopped' ? 'degraded' : 'online';

  const label =
    status === 'running'
      ? 'Pipeline active'
      : status === 'stopped'
        ? 'Halted'
        : status === 'completed'
          ? 'Last run OK'
          : 'System online';

  const dotClass =
    online === 'busy'
      ? 'bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.9)] animate-pulse'
      : online === 'degraded'
        ? 'bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.7)]'
        : 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]';

  return (
    <header className="relative z-20 border-b border-cyan-500/10 bg-slate-950/65 px-4 py-4 backdrop-blur-2xl">
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-cyan-400/25 to-transparent" />
      <div className="mx-auto flex max-w-[1800px] items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="relative flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-400/35 bg-slate-900/90 shadow-[0_0_28px_rgba(34,211,238,0.2),inset_0_0_20px_rgba(59,130,246,0.08)]">
            <Crosshair className="h-6 w-6 text-cyan-400" strokeWidth={1.25} aria-hidden />
          </div>
          <div>
            <h1 className="font-display text-xl font-extrabold tracking-[0.12em] text-transparent [text-shadow:0_0_24px_rgba(34,211,238,0.35)] bg-gradient-to-b from-cyan-200 via-cyan-400 to-blue-600 bg-clip-text md:text-2xl">
              GAMELENS
            </h1>
            <p className="font-data text-[10px] font-medium uppercase tracking-[0.35em] text-blue-500/60">
              Dev build console
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 rounded-2xl border border-slate-800/90 bg-slate-900/50 px-4 py-2.5 backdrop-blur-md">
          <div className={`h-2.5 w-2.5 shrink-0 rounded-full ${dotClass}`} aria-hidden />
          <div className="text-right">
            <p className="font-data text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
              Status
            </p>
            <p className="font-data text-sm font-semibold text-cyan-100/90">{label}</p>
          </div>
          <Activity className="hidden h-4 w-4 text-cyan-500/50 sm:block" aria-hidden />
        </div>
      </div>
    </header>
  );
}
