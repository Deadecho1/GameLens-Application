import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, Sparkles } from 'lucide-react';

/**
 * Success HUD — green flash + banner. Parent clears ui.completionCelebrationActive after delay.
 */
export default function MissionSuccessOverlay({ active }) {
  return (
    <AnimatePresence>
      {active && (
        <>
          <motion.div
            key="success-flash"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0.55, 0.2, 0] }}
            transition={{ duration: 1.2, times: [0, 0.15, 0.4, 1], ease: 'easeOut' }}
            className="pointer-events-none fixed inset-0 z-[90] bg-emerald-400 mix-blend-screen"
            aria-hidden
          />
          <motion.div
            key="success-banner"
            initial={{ y: -120, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -80, opacity: 0 }}
            transition={{ type: 'spring', damping: 22, stiffness: 280, delay: 0.05 }}
            className="pointer-events-none fixed left-1/2 top-[18%] z-[91] w-[min(92vw,520px)] -translate-x-1/2"
          >
            <div className="relative overflow-hidden rounded-2xl border border-emerald-400/50 bg-slate-950/90 px-8 py-5 shadow-[0_0_48px_rgba(16,185,129,0.35),inset_0_1px_0_rgba(52,211,153,0.2)] backdrop-blur-2xl">
              <motion.div
                className="pointer-events-none absolute inset-0 bg-gradient-to-r from-emerald-500/10 via-transparent to-cyan-500/10"
                animate={{ x: ['-100%', '100%'] }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              />
              <div className="relative flex items-center gap-4">
                <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-emerald-400/40 bg-emerald-500/15 text-emerald-400">
                  <Trophy className="h-7 w-7" />
                </div>
                <div>
                  <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-emerald-400/90">
                    Mission accomplished
                  </p>
                  <p className="font-data mt-1 text-sm text-slate-300">
                    Pipeline complete — telemetry unlocked in{' '}
                    <span className="text-cyan-400">ANALYTICS</span>.
                  </p>
                </div>
                <Sparkles className="ml-auto hidden h-6 w-6 text-cyan-400/60 sm:block" />
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
