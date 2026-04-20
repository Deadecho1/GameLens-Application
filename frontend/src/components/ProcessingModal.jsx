import { useRef, useEffect } from 'react';
import {
  X,
  FolderOpen,
  Play,
  Square,
  Trash2,
} from 'lucide-react';

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
 * Large processing workflow modal.
 * Reads/writes processing.* and ui.processingModalOpen.
 *
 * BACKEND:
 * - Replace mock “Choose Folder” with native folder picker + POST path.
 * - Populate processing.videoFiles from scan API.
 * - Stream processing.logs from worker / WebSocket.
 * - Run / Stop → POST job control endpoints; mirror status from server.
 */
export default function ProcessingModal({ data, onPatch, onRun, onStop, onClearLogs, onChooseFolder }) {
  const { setup, processing, ui } = data;
  const logRef = useRef(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [processing.logs]);

  if (!ui.processingModalOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="processing-modal-title"
    >
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b border-slate-800 px-6 py-4">
          <div>
            <h2 id="processing-modal-title" className="text-xl font-bold text-slate-100">
              Processing into version:{' '}
              <span className="text-emerald-400">{setup.selectedVersion}</span>
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Game: <span className="text-slate-400">{setup.selectedGame}</span>
            </p>
          </div>
          <button
            type="button"
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-slate-300"
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

        <div className="flex-1 overflow-y-auto px-6 py-5">
          {/* Pipeline */}
          <section className="mb-6 rounded-2xl border border-slate-800 bg-slate-800/40 p-4">
            <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              Pipeline
            </h3>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 hover:border-emerald-500/40 hover:text-emerald-300"
                onClick={onChooseFolder}
              >
                <FolderOpen className="h-4 w-4" />
                Choose Folder
              </button>
              <code className="flex-1 min-w-[200px] truncate rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-400">
                {processing.pipelinePath}
              </code>
            </div>
          </section>

          {/* File inventory */}
          <section className="mb-6">
            <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              File inventory
            </h3>
            <ul className="max-h-36 overflow-y-auto rounded-2xl border border-slate-800 bg-slate-950/80 p-2 text-sm">
              {processing.videoFiles.length === 0 ? (
                <li className="px-3 py-4 text-center text-slate-600">No videos — choose a folder (mock)</li>
              ) : (
                processing.videoFiles.map((f) => (
                  <li
                    key={f}
                    className="rounded-lg px-3 py-2 text-slate-300 hover:bg-slate-800/80"
                  >
                    {f}
                  </li>
                ))
              )}
            </ul>
          </section>

          {/* Options */}
          <section className="mb-6">
            <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              Options
            </h3>
            <div className="flex flex-wrap gap-4">
              {OPTIONS.map((opt) => (
                <label
                  key={opt.value}
                  className="flex cursor-pointer items-center gap-2 rounded-xl border border-slate-800 bg-slate-800/30 px-4 py-2 text-sm text-slate-300 has-[:checked]:border-blue-500/50 has-[:checked]:bg-blue-500/10"
                >
                  <input
                    type="radio"
                    name="proc-option"
                    className="accent-emerald-500"
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

          {/* Execution */}
          <section className="mb-6 flex flex-wrap items-center gap-4">
            <button
              type="button"
              disabled={processing.status === 'running'}
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-bold text-white shadow-[0_0_20px_rgba(16,185,129,0.25)] hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={onRun}
            >
              <Play className="h-4 w-4 fill-current" />
              Run
            </button>
            <button
              type="button"
              disabled={processing.status !== 'running'}
              className="inline-flex items-center gap-2 rounded-xl bg-red-600 px-5 py-2.5 text-sm font-bold text-white hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-40"
              onClick={onStop}
            >
              <Square className="h-4 w-4 fill-current" />
              Stop
            </button>
            <span className="rounded-full border border-slate-800 bg-slate-800/60 px-4 py-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Status:{' '}
              <span className="text-slate-100">{STATUS_LABEL[processing.status] ?? processing.status}</span>
            </span>
          </section>

          {/* Terminal */}
          <section>
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Console
              </h3>
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded-lg border border-slate-800 px-2 py-1 text-xs font-semibold text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                onClick={onClearLogs}
              >
                <Trash2 className="h-3.5 w-3.5" />
                Clear
              </button>
            </div>
            <pre
              ref={logRef}
              className="max-h-48 overflow-y-auto rounded-2xl border border-slate-800 bg-black p-4 font-mono text-xs leading-relaxed text-emerald-400/95"
            >
              {processing.logs.join('\n')}
            </pre>
          </section>
        </div>
      </div>
    </div>
  );
}
