import { useCallback, useRef, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  SlidersHorizontal,
  Rocket,
  Zap,
  Gamepad2,
  GitBranch,
  Plus,
  RefreshCw,
  UploadCloud,
  FolderOpen,
  Play,
  Square,
  Trash2,
  Film,
  ChevronRight,
} from 'lucide-react';

const OPTIONS = [
  { value: 'only event', label: 'Only event' },
  { value: 'only export', label: 'Only export' },
  { value: 'verbose', label: 'Verbose' },
];

const STEPS = [
  { n: 1, label: 'CONFIGURE', short: 'Configure', icon: SlidersHorizontal },
  { n: 2, label: 'INITIALIZE', short: 'Initialize', icon: Rocket },
  { n: 3, label: 'EXECUTE', short: 'Execute', icon: Zap },
];

const GLOW = {
  blue: 'from-blue-600/25 via-blue-500/5 to-transparent',
  cyan: 'from-cyan-500/20 via-blue-600/5 to-transparent',
};

const STATUS_LABEL = {
  idle: 'READY',
  running: 'RUNNING',
  stopped: 'HALTED',
  completed: 'COMPLETE',
};

/**
 * MISSION START — unified 3-step workflow. Reads ui.workflowStep, setup.*, processing.*.
 */
export default function WorkflowTab({
  data,
  onPatch,
  onAddGame,
  onAddVersion,
  onChooseFolder,
  onRun,
  onStop,
  onClearLogs,
}) {
  const { setup, processing, ui } = data;
  const step = ui.workflowStep;
  const logRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const logCount = processing.logs.length;

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [processing.logs, step]);

  const ingestFiles = useCallback(
    (fileList) => {
      const files = Array.from(fileList || []).filter(
        (f) => f.type.startsWith('video/') || /\.(mp4|webm|mov|mkv|avi)$/i.test(f.name)
      );
      if (files.length === 0) return;
      const names = files.map((f) => f.name);
      onPatch({
        processing: {
          ...processing,
          videoFiles: [...new Set([...processing.videoFiles, ...names])],
          pipelinePath: `LOCAL_STAGING://${names[0]}`,
        },
      });
    },
    [onPatch, processing]
  );

  const progressVisual = useMemo(() => {
    const s = processing.status;
    if (s === 'completed') return { width: '100%', variant: 'done' };
    if (s === 'stopped') return { width: '32%', variant: 'warn' };
    if (s === 'running') return { width: null, variant: 'pulse' };
    return { width: '12%', variant: 'idle' };
  }, [processing.status]);

  const configureReady =
    Boolean(setup.selectedGame?.trim()) && Boolean(setup.selectedVersion?.trim());

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="mx-auto max-w-6xl px-4 py-8 md:py-10"
    >
      <header className="mb-8 text-center md:text-left">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Mission start
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Workflow sequence
        </h2>
        <p className="font-data mt-2 text-sm text-slate-500">
          Step <span className="text-cyan-400">{step}</span> of 3 · synced to{' '}
          <code className="rounded bg-slate-900 px-1 text-cyan-700/90">ui.workflowStep</code>
        </p>
      </header>

      {/* Top timeline */}
      <div className="mb-10 rounded-2xl border border-slate-800/90 bg-slate-950/50 p-4 backdrop-blur-xl md:p-6">
        <div className="flex items-start justify-between gap-2 md:gap-4">
          {STEPS.map((s, idx) => {
            const Icon = s.icon;
            const active = step === s.n;
            const done = step > s.n;
            const isLast = idx === STEPS.length - 1;
            return (
              <div key={s.n} className="flex min-w-0 flex-1 items-start">
                <div className="flex w-full flex-col items-center">
                  <motion.div
                    className={`relative z-[1] flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border-2 transition md:h-14 md:w-14 ${
                      active
                        ? 'border-cyan-400 bg-cyan-500/15 text-cyan-300 shadow-[0_0_24px_rgba(34,211,238,0.35)]'
                        : done
                          ? 'border-blue-500/50 bg-blue-500/10 text-blue-300'
                          : 'border-slate-800 bg-slate-900/80 text-slate-600'
                    }`}
                    animate={active ? { scale: [1, 1.04, 1] } : {}}
                    transition={{ duration: 2, repeat: active ? Infinity : 0, ease: 'easeInOut' }}
                  >
                    <Icon className="h-5 w-5 md:h-6 md:w-6" strokeWidth={1.25} />
                  </motion.div>
                  <p className="font-display mt-2 hidden text-center text-[9px] font-bold tracking-wider text-slate-500 sm:block md:text-[10px]">
                    {s.label}
                  </p>
                  <p className="font-display mt-1 text-center text-[8px] font-bold tracking-wider text-slate-600 sm:hidden">
                    {s.short}
                  </p>
                </div>
                {!isLast && (
                  <div className="relative mx-1 mt-6 h-0.5 min-w-[12px] flex-1 self-start overflow-hidden rounded-full bg-slate-800 md:mx-2 md:mt-7">
                    <motion.div
                      className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-500 shadow-[0_0_12px_rgba(34,211,238,0.5)]"
                      initial={false}
                      animate={{ width: step > s.n ? '100%' : '0%' }}
                      transition={{ type: 'spring', stiffness: 200, damping: 26 }}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div
            key="wf-1"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.25 }}
          >
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
            <div className="mt-10 flex justify-center">
              <div className="gl-radial-glow relative inline-block rounded-2xl p-[1px]">
                <motion.button
                  type="button"
                  disabled={!configureReady}
                  className="relative z-[1] inline-flex items-center gap-2 rounded-2xl border border-cyan-500/50 bg-slate-950 px-8 py-4 font-display text-xs font-bold uppercase tracking-[0.2em] text-cyan-200 shadow-[0_0_32px_rgba(34,211,238,0.25)] transition hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                  whileHover={configureReady ? { scale: 1.02 } : {}}
                  whileTap={configureReady ? { scale: 0.98 } : {}}
                  onClick={() =>
                    configureReady &&
                    onPatch({ ui: { ...ui, workflowStep: 2 } })
                  }
                >
                  Proceed to mission setup
                  <ChevronRight className="h-4 w-4" />
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {step === 2 && (
          <motion.div
            key="wf-2"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.25 }}
            className="grid gap-8 lg:grid-cols-2"
          >
            <section>
              <h3 className="font-display mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-500/80">
                Ingest port
              </h3>
              <div
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
                }}
                onDragEnter={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false);
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  ingestFiles(e.dataTransfer.files);
                }}
                onClick={() => inputRef.current?.click()}
                className={`relative cursor-pointer rounded-2xl border-2 border-dashed px-6 py-12 text-center transition md:py-14 ${
                  dragOver
                    ? 'border-cyan-400 bg-cyan-500/10'
                    : 'border-blue-500/35 bg-slate-950/40 gl-upload-pulse hover:border-cyan-400/45'
                }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="video/*,.mp4,.webm,.mov,.mkv,.avi"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    ingestFiles(e.target.files);
                    e.target.value = '';
                  }}
                />
                <UploadCloud className="mx-auto h-10 w-10 text-cyan-400/90" strokeWidth={1.15} />
                <p className="font-display mt-4 text-sm font-bold uppercase tracking-wider text-slate-200">
                  Drop video clip
                </p>
                <p className="font-data mt-2 text-xs text-slate-500">
                  Writes to <code className="text-cyan-700/80">processing.videoFiles</code>
                </p>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className="font-data inline-flex items-center gap-2 rounded-xl border border-blue-500/35 bg-slate-900/60 px-4 py-2 text-sm text-cyan-200 backdrop-blur-sm transition hover:border-cyan-400/50"
                  onClick={onChooseFolder}
                >
                  <FolderOpen className="h-4 w-4" />
                  Pipeline path (mock)
                </button>
                <code className="font-data max-w-full flex-1 truncate rounded-lg border border-slate-800 bg-black/50 px-3 py-2 text-[11px] text-cyan-600/80">
                  {processing.pipelinePath}
                </code>
              </div>
              <ul className="font-data mt-4 max-h-28 space-y-1 overflow-y-auto rounded-xl border border-slate-800 bg-black/35 p-2 text-xs">
                {processing.videoFiles.length === 0 ? (
                  <li className="py-4 text-center text-slate-600">No clips staged</li>
                ) : (
                  processing.videoFiles.map((f) => (
                    <li key={f} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-cyan-100/80">
                      <Film className="h-3.5 w-3.5 shrink-0 text-blue-400/70" />
                      {f}
                    </li>
                  ))
                )}
              </ul>
            </section>

            <section className="flex flex-col">
              <h3 className="font-display mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-500/80">
                Trace flags
              </h3>
              <div className="flex flex-wrap gap-2">
                {OPTIONS.map((opt) => (
                  <label
                    key={opt.value}
                    className="font-data flex cursor-pointer items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/50 px-3 py-2 text-xs text-slate-400 backdrop-blur-sm has-checked:border-cyan-500/40 has-checked:bg-cyan-500/5 has-checked:text-cyan-200"
                  >
                    <input
                      type="radio"
                      name="workflow-proc-opt"
                      className="accent-cyan-500"
                      checked={processing.selectedOption === opt.value}
                      onChange={() =>
                        onPatch({
                          processing: { ...processing, selectedOption: opt.value },
                        })
                      }
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
              <p className="font-data mt-6 text-xs text-slate-600">
                Mission: <span className="text-cyan-500/80">{setup.selectedGame}</span> · Build:{' '}
                <span className="text-blue-400/80">{setup.selectedVersion}</span>
              </p>
              <div className="mt-auto flex flex-col gap-3 pt-8">
                <button
                  type="button"
                  className="font-display rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 text-[11px] font-bold uppercase tracking-wider text-slate-400 transition hover:border-slate-600 hover:text-slate-200"
                  onClick={() => onPatch({ ui: { ...ui, workflowStep: 1 } })}
                >
                  ← Back to configure
                </button>
                <div className="gl-radial-glow relative">
                  <motion.button
                    type="button"
                    className="relative z-[1] w-full rounded-2xl border border-blue-500/45 bg-gradient-to-b from-blue-600 to-blue-800 py-4 font-display text-sm font-bold uppercase tracking-[0.15em] text-white shadow-[0_0_32px_rgba(59,130,246,0.3)]"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    onClick={() => onPatch({ ui: { ...ui, workflowStep: 3 } })}
                  >
                    Ready for launch
                  </motion.button>
                </div>
              </div>
            </section>
          </motion.div>
        )}

        {step === 3 && (
          <motion.div
            key="wf-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.25 }}
          >
            <div className="mb-6 rounded-2xl border border-slate-800/90 bg-slate-900/45 p-4 backdrop-blur-md">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <span className="font-data text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  Engine state
                </span>
                <span className="font-data text-xs font-bold tracking-wider text-cyan-400">
                  {STATUS_LABEL[processing.status]}
                </span>
              </div>
              <div className="relative h-2 overflow-hidden rounded-full bg-black/50 ring-1 ring-slate-800">
                {progressVisual.variant === 'pulse' ? (
                  <motion.div
                    className="gl-neon-bar-fill absolute top-0 h-full rounded-full"
                    initial={{ left: '0%', width: '38%' }}
                    animate={{
                      left: ['0%', '52%', '8%', '44%'],
                      width: ['36%', '44%', '40%', '38%'],
                    }}
                    transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
                  />
                ) : (
                  <motion.div
                    className={`h-full rounded-full ${
                      progressVisual.variant === 'warn'
                        ? 'bg-gradient-to-r from-amber-600 to-red-600 shadow-[0_0_16px_rgba(239,68,68,0.35)]'
                        : progressVisual.variant === 'done'
                          ? 'gl-neon-bar-fill'
                          : 'bg-slate-800'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: progressVisual.width }}
                    transition={{ type: 'spring', stiffness: 120, damping: 18 }}
                  />
                )}
              </div>
            </div>

            <div className="flex flex-col gap-4 sm:flex-row">
              <div className="gl-radial-glow relative flex-1">
                <motion.button
                  type="button"
                  disabled={processing.status === 'running'}
                  className="relative z-[1] flex w-full items-center justify-center gap-3 rounded-2xl border border-emerald-500/40 bg-gradient-to-b from-emerald-600 to-emerald-800 py-5 font-display text-sm font-bold uppercase tracking-[0.18em] text-white shadow-[0_0_36px_rgba(16,185,129,0.3)] disabled:cursor-not-allowed disabled:opacity-45"
                  whileHover={processing.status !== 'running' ? { scale: 1.01 } : {}}
                  whileTap={processing.status !== 'running' ? { scale: 0.99 } : {}}
                  onClick={onRun}
                >
                  <Play className="h-6 w-6 fill-current" />
                  Run GameLens
                </motion.button>
              </div>
              <button
                type="button"
                disabled={processing.status !== 'running'}
                className="flex flex-1 items-center justify-center gap-3 rounded-2xl border border-red-500/45 bg-gradient-to-b from-red-600 to-red-800 py-5 font-display text-sm font-bold uppercase tracking-[0.18em] text-white shadow-[0_0_28px_rgba(239,68,68,0.2)] disabled:cursor-not-allowed disabled:opacity-35"
                onClick={onStop}
              >
                <Square className="h-6 w-6 fill-current" />
                Stop
              </button>
            </div>

            <button
              type="button"
              className="font-data mt-4 text-xs text-slate-500 underline-offset-4 hover:text-slate-400 hover:underline"
              onClick={() => onPatch({ ui: { ...ui, workflowStep: 2 } })}
            >
              ← Back to initialize
            </button>

            <section className="mt-10">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
                  Diagnostics terminal
                </h3>
                <button
                  type="button"
                  className="font-data inline-flex items-center gap-1 rounded-lg border border-slate-800 px-2 py-1 text-[11px] font-semibold text-slate-500 transition hover:border-cyan-500/30 hover:text-cyan-400"
                  onClick={onClearLogs}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Clear
                </button>
              </div>
              <div
                ref={logRef}
                className="gl-terminal-scanlines max-h-72 min-h-[220px] overflow-y-auto rounded-2xl border border-cyan-500/15 bg-black/80 p-4 font-data text-xs leading-relaxed text-emerald-400/95 shadow-[inset_0_0_48px_rgba(34,211,238,0.04)] backdrop-blur-sm"
              >
                {processing.logs.length === 0 ? (
                  <span className="text-slate-600">&gt; buffer empty</span>
                ) : (
                  processing.logs.map((line, i) => (
                    <motion.div
                      key={`${i}-${line.slice(0, 32)}`}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.18 }}
                      className={`whitespace-pre-wrap border-l-2 border-transparent pl-2 ${
                        i === logCount - 1 ? 'gl-log-line-new border-cyan-500/40 text-cyan-200' : ''
                      }`}
                    >
                      <span className="text-cyan-700/90">&gt; </span>
                      {line}
                    </motion.div>
                  ))
                )}
              </div>
            </section>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function SelectionCard({ icon: Icon, kicker, value, onChange, onAdd, glowVariant }) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6 shadow-[inset_0_1px_0_rgba(34,211,238,0.06)] backdrop-blur-md transition hover:border-cyan-500/25">
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
