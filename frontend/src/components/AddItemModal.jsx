import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

/**
 * Register modal — writes setup.* via parent onConfirm; ui drafts in dataStore.
 * BACKEND: POST new mission/build then merge API response.
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
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="add-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] bg-slate-950/75 backdrop-blur-md"
            role="presentation"
            onClick={onClose}
          />
          <motion.div
            key="add-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby={inputId + '-title'}
            initial={{ opacity: 0, scale: 0.94, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 8 }}
            transition={{ type: 'spring', damping: 24, stiffness: 320 }}
            className="fixed inset-0 z-[61] flex items-center justify-center p-4 pointer-events-none"
          >
            <div
              className="pointer-events-auto w-full max-w-md rounded-2xl border border-cyan-500/20 bg-slate-950/95 p-6 shadow-[0_0_48px_rgba(59,130,246,0.12)] backdrop-blur-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="mb-4 flex items-start justify-between gap-2">
                <h2
                  id={inputId + '-title'}
                  className="font-display text-lg font-bold uppercase tracking-wide text-slate-100"
                >
                  {title}
                </h2>
                <button
                  type="button"
                  className="rounded-lg p-1 text-slate-300 transition hover:bg-slate-800 hover:text-cyan-400"
                  onClick={onClose}
                  aria-label="Close"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <input
                id={inputId}
                type="text"
                className="font-data mb-6 w-full rounded-xl border border-slate-800 bg-black/50 px-4 py-3 text-slate-100 outline-none ring-cyan-500/20 placeholder:text-slate-400 focus:ring-2"
                placeholder="Identifier…"
                value={draftValue}
                onChange={(e) => onDraftChange(e.target.value)}
                autoFocus
              />
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  className="font-data rounded-xl px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-slate-800 hover:text-slate-300"
                  onClick={onClose}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="font-display rounded-xl bg-blue-600 px-4 py-2 text-sm font-bold uppercase tracking-wider text-white shadow-[0_0_20px_rgba(59,130,246,0.3)] hover:bg-blue-500 disabled:opacity-40"
                  disabled={!draftValue.trim()}
                  onClick={onConfirm}
                >
                  Commit
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
