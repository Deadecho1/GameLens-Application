import { X } from 'lucide-react';

/**
 * Centered “Add New” modal for games or versions.
 * Writes to setup.* and ui.* only.
 * BACKEND: On confirm, POST new game/version then merge server response into setup.
 */
export default function AddItemModal({
  open,
  title,
  draftValue,
  onDraftChange,
  onClose,
  onConfirm,
  inputId,
}) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby={inputId + '-title'}
    >
      <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-2xl shadow-black/50">
        <div className="mb-4 flex items-start justify-between gap-2">
          <h2 id={inputId + '-title'} className="text-lg font-bold text-slate-100">
            {title}
          </h2>
          <button
            type="button"
            className="rounded-lg p-1 text-slate-500 hover:bg-slate-800 hover:text-slate-300"
            onClick={onClose}
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <input
          id={inputId}
          type="text"
          className="mb-6 w-full rounded-xl border border-slate-800 bg-slate-800/80 px-4 py-3 text-slate-100 outline-none ring-blue-500/30 placeholder:text-slate-600 focus:ring-2"
          placeholder="Name…"
          value={draftValue}
          onChange={(e) => onDraftChange(e.target.value)}
          autoFocus
        />
        <div className="flex justify-end gap-2">
          <button
            type="button"
            className="rounded-xl px-4 py-2 text-sm font-semibold text-slate-400 hover:bg-slate-800"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-bold text-white hover:bg-blue-500 disabled:opacity-40"
            disabled={!draftValue.trim()}
            onClick={onConfirm}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
