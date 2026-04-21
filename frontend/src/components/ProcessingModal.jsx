import { useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, FolderOpen, Play, Square, Trash2, Activity } from 'lucide-react';

const OPTIONS = [
  { value: 'only event', label: 'Only event' },
  { value: 'only export', label: 'Only export' },
  { value: 'verbose', label: 'Verbose' },
];

const STATUS_LABEL = {
  idle: 'Ready',
  running: 'Running…',
  stopped: 'Stopped',
  completed: 'Completed',
};

/**
 * Diagnostics “engine” modal — terminal aesthetic, neon status, scanline log stack.
 * BACKEND: stream logs into processing.logs; drive processing.status from job worker.
 */
export default function ProcessingModal({ data, onPatch, onRun, onStop, onClearLogs, onChooseFolder }) {
  const { setup, processing, ui } = data;
  const logRef = useRef(null);
  const logCount = processing.logs.length;

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [processing.logs]);

  const progressVisual = useMemo(() => {
    const s = processing.status;
    if (s === 'completed') return { width: '100%', variant: 'done' };
    if (s === 'stopped') return { width: '32%', variant: 'warn' };
    if (s === 'running') return { width: null, variant: 'pulse' };
    return { width: '12%', variant: 'idle' };
  }, [processing.status]);

  return (
    <AnimatePresence>
      {ui.processingModalOpen && (
        <>
          <motion.div
            key="proc-backdrop"
            role="presentation"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-lg"
            onClick={() =>
              onPatch({
                ui: { ...ui, processingModalOpen: false },
              })
            }
          />
          <motion.div
            key="proc-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="processing-modal-title"
            initial={{ opacity: 0, scale: 0.96, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 16 }}
            transition={{ type: 'spring', damping: 26, stiffness: 320 }}
            className="fixed inset-0 z-[51] flex items-center justify-center p-4 pointer-events-none"
          >
            <div
              className="pointer-events-auto flex max-h-[92vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-cyan-500/20 bg-slate-950/95 shadow-[0_0_60px_rgba(59,130,246,0.15),inset_0_1px_0_rgba(34,211,238,0.08)] backdrop-blur-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative border-b border-slate-800/90 px-6 py-4">
                <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_80%_120%_at_50%_-20%,rgba(59,130,246,0.12),transparent)]" />
                <div className="relative flex items-start justify-between gap-4">
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 flex h-10 w-10 items-center justify-center rounded-lg border border-blue-500/30 bg-blue-500/10 text-cyan-400">
                      <Activity className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="font-data text-[10px] font-semibold uppercase tracking-[0.25em] text-blue-500/80">
                        Diagnostics terminal
                      </p>
                      <h2 id="processing-modal-title" className="font-display text-lg font-bold text-slate-100 md:text-xl">
                        Pipeline ·{' '}
                        <span className="text-cyan-400">{setup.selectedVersion}</span>
                      </h2>
                      <p className="font-data mt-1 text-xs text-slate-500">
                        Mission <span className="text-slate-400">{setup.selectedGame}</span>
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-800 hover:text-cyan-400"
                    onClick={() =>
                      onPatch({
                        ui: { ...ui, processingModalOpen: false },
                      })
                    }
                    aria-label="Close modal"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>

                {/* Neon status bar */}
                <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-slate-900 ring-1 ring-slate-800">
                  {progressVisual.variant === 'pulse' ? (
                    <motion.div
                      className="gl-neon-bar-fill absolute top-0 h-full rounded-full"
                      initial={{ left: '0%', width: '38%' }}
                      animate={{ left: ['0%', '55%', '5%', '40%'], width: ['38%', '42%', '48%', '36%'] }}
                      transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  ) : (
                    <motion.div
                      className={`h-full rounded-full ${
                        progressVisual.variant === 'warn'
                          ? 'bg-gradient-to-r from-amber-600 to-red-600 shadow-[0_0_16px_rgba(239,68,68,0.4)]'
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
                <p className="font-data mt-2 text-[10px] uppercase tracking-wider text-slate-500">
                  Subsystem: <span className="text-cyan-400/90">{STATUS_LABEL[processing.status]}</span>
                </p>
              </div>

              <div className="flex-1 overflow-y-auto px-6 py-5">
                <section className="mb-6 rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
                  <h3 className="font-display mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
                    Data path
                  </h3>
                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      className="font-data inline-flex items-center gap-2 rounded-xl border border-blue-500/30 bg-slate-900 px-4 py-2 text-sm font-semibold text-cyan-200 transition hover:border-cyan-400/50 hover:shadow-[0_0_20px_rgba(59,130,246,0.15)]"
                      onClick={onChooseFolder}
                    >
                      <FolderOpen className="h-4 w-4" />
                      Choose folder
                    </button>
                    <code className="font-data min-w-[200px] flex-1 truncate rounded-lg border border-slate-800 bg-black/60 px-3 py-2 text-xs text-cyan-500/70">
                      {processing.pipelinePath}
                    </code>
                  </div>
                </section>

                <section className="mb-6">
                  <h3 className="font-display mb-2 text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
                    Video inventory
                  </h3>
                  <ul className="font-data max-h-36 overflow-y-auto rounded-xl border border-slate-800 bg-black/40 p-2 text-sm">
                    {processing.videoFiles.length === 0 ? (
                      <li className="px-3 py-6 text-center text-slate-600">No assets staged</li>
                    ) : (
                      processing.videoFiles.map((f) => (
                        <li
                          key={f}
                          className="rounded-lg px-3 py-2 text-cyan-100/80 hover:bg-slate-800/50"
                        >
                          {f}
                        </li>
                      ))
                    )}
                  </ul>
                </section>

                <section className="mb-6">
                  <h3 className="font-display mb-2 text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
                    Trace flags
                  </h3>
                  <div className="flex flex-wrap gap-3">
                    {OPTIONS.map((opt) => (
                      <label
                        key={opt.value}
                        className="font-data flex cursor-pointer items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-2 text-sm text-slate-300 has-checked:border-cyan-500/40 has-checked:bg-cyan-500/5 has-checked:text-cyan-200"
                      >
                        <input
                          type="radio"
                          name="proc-option"
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
                </section>

                <section className="mb-6 flex flex-wrap items-center gap-4">
                  <div className="gl-radial-glow relative inline-block rounded-xl">
                    <button
                      type="button"
                      disabled={processing.status === 'running'}
                      className="relative z-[1] inline-flex items-center gap-2 rounded-xl border border-emerald-500/40 bg-emerald-600 px-5 py-2.5 font-display text-xs font-bold uppercase tracking-wider text-white shadow-[0_0_24px_rgba(16,185,129,0.25)] disabled:cursor-not-allowed disabled:opacity-45"
                      onClick={onRun}
                    >
                      <Play className="h-4 w-4 fill-current" />
                      Run
                    </button>
                  </div>
                  <button
                    type="button"
                    disabled={processing.status !== 'running'}
                    className="inline-flex items-center gap-2 rounded-xl border border-red-500/40 bg-red-600/90 px-5 py-2.5 font-display text-xs font-bold uppercase tracking-wider text-white hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-35"
                    onClick={onStop}
                  >
                    <Square className="h-4 w-4 fill-current" />
                    Stop
                  </button>
                </section>

                <section>
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
                      System log
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
                    className="gl-terminal-scanlines max-h-52 overflow-y-auto rounded-xl border border-cyan-500/15 bg-black p-4 font-data text-xs leading-relaxed text-emerald-400/95 shadow-[inset_0_0_40px_rgba(34,211,238,0.04)]"
                  >
                    {processing.logs.length === 0 ? (
                      <span className="text-slate-600">&gt; awaiting signal…</span>
                    ) : (
                      processing.logs.map((line, i) => (
                        <motion.div
                          key={`${i}-${line.slice(0, 24)}`}
                          initial={{ opacity: 0, x: -6 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.2 }}
                          className={`whitespace-pre-wrap border-l-2 border-transparent pl-2 ${
                            i === logCount - 1 ? 'gl-log-line-new border-cyan-500/40 text-cyan-200' : ''
                          }`}
                        >
                          <span className="text-cyan-600/80">&gt; </span>
                          {line}
                        </motion.div>
                      ))
                    )}
                  </div>
                  <p className="font-data mt-2 text-[10px] text-slate-600">
                    BOOT_SEQ: new lines animate in; BACKEND append to{' '}
                    <code className="text-slate-500">processing.logs[]</code>
                  </p>
                </section>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
