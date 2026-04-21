import { motion } from 'framer-motion';
import { Gamepad2, GitBranch, Plus, RefreshCw } from 'lucide-react';

/**
 * SETUP — landing zone: selection cards + add-new. All state from dataStore.setup / ui.
 */
export default function SetupTab({ data, onPatch, onAddGame, onAddVersion }) {
  const { setup, ui } = data;

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 12 }}
      transition={{ duration: 0.25 }}
      className="mx-auto max-w-5xl px-4 py-10 md:py-14"
    >
      <header className="mb-10 text-center md:text-left">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Landing zone
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Configure session
        </h2>
        <p className="font-data mt-2 max-w-xl text-sm text-slate-500">
          Lock mission and build before ingest. Values sync to{' '}
          <code className="rounded bg-slate-900 px-1 text-cyan-600/80">setup.*</code>.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        <SelectionCard
          icon={Gamepad2}
          kicker="Mission"
          value={setup.selectedGame}
          onChange={() => onPatch({ ui: { ...ui, changePicker: 'game' } })}
          onAdd={onAddGame}
          glowVariant="blue"
        />
        <SelectionCard
          icon={GitBranch}
          kicker="Build"
          value={setup.selectedVersion}
          onChange={() => onPatch({ ui: { ...ui, changePicker: 'version' } })}
          onAdd={onAddVersion}
          glowVariant="cyan"
        />
      </div>
    </motion.div>
  );
}

const GLOW = {
  blue: 'from-blue-600/25 via-blue-500/5 to-transparent',
  cyan: 'from-cyan-500/20 via-blue-600/5 to-transparent',
};

function SelectionCard({ icon: Icon, kicker, value, onChange, onAdd, glowVariant }) {
  return (
    <div
      className={`group relative overflow-hidden rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6 shadow-[inset_0_1px_0_rgba(34,211,238,0.06)] backdrop-blur-md transition hover:border-cyan-500/25`}
    >
      <div
        className={`pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-gradient-to-br ${GLOW[glowVariant]} blur-3xl opacity-80`}
      />
      <div className="relative flex items-start justify-between gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/20 bg-slate-950/60 text-cyan-400">
          <Icon className="h-6 w-6" strokeWidth={1.25} />
        </div>
        <button
          type="button"
          title="Add new"
          className="flex h-11 w-11 items-center justify-center rounded-full border border-cyan-400/30 bg-slate-950/80 text-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.25)] transition hover:scale-105 hover:border-cyan-400/60 hover:shadow-[0_0_28px_rgba(34,211,238,0.4)]"
          onClick={onAdd}
        >
          <Plus className="h-5 w-5" strokeWidth={2.5} />
        </button>
      </div>
      <p className="font-display relative mt-6 text-[10px] font-bold uppercase tracking-[0.25em] text-slate-500">
        {kicker}
      </p>
      <p className="font-data relative mt-2 text-lg font-semibold text-cyan-100/95 md:text-xl">
        {value}
      </p>
      <button
        type="button"
        className="font-display relative mt-6 inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-2.5 text-[11px] font-bold uppercase tracking-wider text-slate-300 backdrop-blur-sm transition hover:border-blue-500/40 hover:text-cyan-200"
        onClick={onChange}
      >
        <RefreshCw className="h-3.5 w-3.5" />
        Change
      </button>
    </div>
  );
}
