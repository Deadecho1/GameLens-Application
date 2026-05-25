import { useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, Check, Settings, Trash2, X } from 'lucide-react';
import RunSessionAnalytics from './analytics/runSession/RunSessionAnalytics';

/**
 * Full-screen post-processing review — blocks backdrop dismiss; footer actions only.
 */
export default function PostProcessingReviewModal({
  open,
  data,
  pendingRun,
  onDiscard,
  onConfirm,
  onGoToTuning,
  onClose,
}) {
  const reviewData = useMemo(() => {
    const history = pendingRun ? [pendingRun] : [];
    return {
      ...data,
      dashboard: {
        ...data.dashboard,
        runsHistory: history,
      },
    };
  }, [data, pendingRun]);

  if (typeof document === 'undefined') return null;

  return createPortal(
    <AnimatePresence>
      {open ? (
        <motion.div
          key="post-processing-review"
          role="dialog"
          aria-modal="true"
          aria-labelledby="post-processing-review-title"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.22 }}
          className="fixed inset-0 z-[9999] flex flex-col bg-slate-950/90 backdrop-blur-md"
        >
          <button
            type="button"
            onClick={onClose}
            className="absolute right-4 top-4 z-10 rounded-lg border border-slate-800 bg-slate-900/90 p-2 text-slate-400 transition hover:border-slate-600 hover:text-slate-200 md:right-6 md:top-6"
            aria-label="Close review without syncing"
          >
            <X className="h-5 w-5" strokeWidth={1.75} aria-hidden />
          </button>
          <header className="relative shrink-0 border-b border-slate-800/90 bg-slate-950/80 px-4 py-4 pr-14 md:px-8 md:pr-16">
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-cyan-500/70">
              Mission complete
            </p>
            <h2
              id="post-processing-review-title"
              className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl"
            >
              Post-processing review
            </h2>
            <p className="font-data mt-2 max-w-3xl text-sm text-slate-400">
              Validate the latest run analysis before syncing to your library. Discard if the model
              misread the session, or open Tuning to correct the pipeline.
            </p>
            {!pendingRun ? (
              <p className="font-data mt-3 flex items-center gap-2 text-xs text-amber-400/90">
                <AlertCircle className="h-4 w-4 shrink-0" aria-hidden />
                Run payload still loading — you can discard or confirm once data appears.
              </p>
            ) : null}
          </header>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
            <RunSessionAnalytics
              data={reviewData}
              embedded
              initialSelectedRunId={pendingRun?.id ?? null}
            />
          </div>

          <footer className="shrink-0 border-t border-slate-800/90 bg-slate-950/95 px-4 py-4 backdrop-blur-xl md:px-8">
            <div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-end gap-3">
              <button
                type="button"
                onClick={onDiscard}
                className="inline-flex items-center gap-2 rounded-lg border border-red-500/45 bg-red-950/50 px-4 py-2.5 font-display text-[10px] font-bold uppercase tracking-[0.14em] text-red-200 transition hover:border-red-400/70 hover:bg-red-900/60"
              >
                <Trash2 className="h-4 w-4 shrink-0" strokeWidth={1.75} aria-hidden />
                Discard analysis
              </button>
              <button
                type="button"
                onClick={onGoToTuning}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-transparent px-4 py-2.5 font-display text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400 transition hover:border-slate-600 hover:bg-slate-900/60 hover:text-slate-200"
              >
                <Settings className="h-4 w-4 shrink-0 opacity-80" strokeWidth={1.75} aria-hidden />
                Model inaccurate? Go to tuning
              </button>
              <button
                type="button"
                onClick={onConfirm}
                className="inline-flex items-center gap-2 rounded-lg border border-cyan-400/55 bg-cyan-500/15 px-5 py-2.5 font-display text-[10px] font-bold uppercase tracking-[0.14em] text-cyan-100 shadow-[0_0_20px_rgba(34,211,238,0.2)] transition hover:border-cyan-300/70 hover:bg-cyan-500/25"
              >
                <Check className="h-4 w-4 shrink-0" strokeWidth={2} aria-hidden />
                Confirm &amp; sync
              </button>
            </div>
          </footer>
        </motion.div>
      ) : null}
    </AnimatePresence>,
    document.body,
  );
}
