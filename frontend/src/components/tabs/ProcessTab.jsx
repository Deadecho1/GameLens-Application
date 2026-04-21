import { useCallback, useRef, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  UploadCloud,
  FolderOpen,
  Play,
  Square,
  Trash2,
  Film,
} from 'lucide-react';

const OPTIONS = [
  { value: 'only event', label: 'Only event' },
  { value: 'only export', label: 'Only export' },
  { value: 'verbose', label: 'Verbose' },
];

const STATUS_LABEL = {
  idle: 'READY',
  running: 'RUNNING',
  stopped: 'HALTED',
  completed: 'COMPLETE',
};

/**
 * PROCESS — ingest + tactical controls + embedded diagnostics terminal.
 * Reads/writes processing.*; displays setup context read-only.
 */
export default function ProcessTab({
  data,
  onPatch,
  onChooseFolder,
  onRun,
  onStop,
  onClearLogs,
}) {
  const { setup, processing } = data;
  const logRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const logCount = processing.logs.length;

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [processing.logs]);

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

  return (
    <motion.div
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -12 }}
      transition={{ duration: 0.25 }}
      className="mx-auto max-w-6xl px-4 py-8 md:py-10"
    >
      <header className="mb-8">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Active run
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Ingest &amp; pipeline
        </h2>
        <p className="font-data mt-2 text-sm text-slate-500">
          <span className="text-cyan-500/80">{setup.selectedGame}</span>
          <span className="mx-2 text-slate-700">·</span>
          <span className="text-blue-400/80">{setup.selectedVersion}</span>
        </p>
      </header>

      {/* Status strip */}
      <div className="mb-8 rounded-2xl border border-slate-800/90 bg-slate-900/45 p-4 backdrop-blur-md">
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

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Upload */}
        <section>
          <h3 className="font-display mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-500/80">
            Upload port
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
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed px-6 py-14 text-center transition md:py-16 ${
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
            <p className="font-data mt-2 text-xs text-slate-500">or click to browse · merges into processing.videoFiles</p>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              className="font-data inline-flex items-center gap-2 rounded-xl border border-blue-500/35 bg-slate-900/60 px-4 py-2 text-sm text-cyan-200 backdrop-blur-sm transition hover:border-cyan-400/50"
              onClick={onChooseFolder}
            >
              <FolderOpen className="h-4 w-4" />
              Cycle folder (mock)
            </button>
            <code className="font-data max-w-full flex-1 truncate rounded-lg border border-slate-800 bg-black/50 px-3 py-2 text-[11px] text-cyan-600/80">
              {processing.pipelinePath}
            </code>
          </div>

          <ul className="font-data mt-4 max-h-32 space-y-1 overflow-y-auto rounded-xl border border-slate-800 bg-black/35 p-2 text-xs">
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

        {/* Controls + options */}
        <section className="flex flex-col gap-6">
          <div>
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
                    name="proc-opt"
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
          </div>

          <div className="mt-auto flex flex-col gap-4 sm:flex-row">
            <div className="gl-radial-glow relative flex-1">
              <button
                type="button"
                disabled={processing.status === 'running'}
                className="relative z-[1] flex w-full items-center justify-center gap-3 rounded-2xl border border-emerald-500/40 bg-gradient-to-b from-emerald-600 to-emerald-800 py-4 font-display text-sm font-bold uppercase tracking-[0.15em] text-white shadow-[0_0_32px_rgba(16,185,129,0.25)] disabled:cursor-not-allowed disabled:opacity-45"
                onClick={onRun}
              >
                <Play className="h-5 w-5 fill-current" />
                Run GameLens
              </button>
            </div>
            <button
              type="button"
              disabled={processing.status !== 'running'}
              className="flex flex-1 items-center justify-center gap-3 rounded-2xl border border-red-500/45 bg-gradient-to-b from-red-600 to-red-800 py-4 font-display text-sm font-bold uppercase tracking-[0.15em] text-white shadow-[0_0_24px_rgba(239,68,68,0.2)] disabled:cursor-not-allowed disabled:opacity-35"
              onClick={onStop}
            >
              <Square className="h-5 w-5 fill-current" />
              Stop
            </button>
          </div>
        </section>
      </div>

      {/* Embedded terminal */}
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
          className="gl-terminal-scanlines max-h-64 min-h-[200px] overflow-y-auto rounded-2xl border border-cyan-500/15 bg-black/80 p-4 font-data text-xs leading-relaxed text-emerald-400/95 shadow-[inset_0_0_48px_rgba(34,211,238,0.04)] backdrop-blur-sm"
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
  );
}
