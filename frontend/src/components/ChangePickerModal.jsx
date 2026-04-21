import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

/**
 * List picker for SETUP “Change” — sets setup.selectedGame or selectedVersion from store lists.
 */
export default function ChangePickerModal({ data, onPatch }) {
  const { setup, ui } = data;
  const kind = ui.changePicker;
  const open = kind === 'game' || kind === 'version';

  const items = kind === 'game' ? setup.games : kind === 'version' ? setup.versions : [];
  const title = kind === 'game' ? 'Select mission' : kind === 'version' ? 'Select build' : '';

  const select = (value) => {
    if (kind === 'game') {
      onPatch({
        setup: { ...setup, selectedGame: value },
        ui: { ...ui, changePicker: null },
      });
    } else if (kind === 'version') {
      onPatch({
        setup: { ...setup, selectedVersion: value },
        ui: { ...ui, changePicker: null },
      });
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="pick-back"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[70] bg-slate-950/80 backdrop-blur-lg"
            onClick={() => onPatch({ ui: { ...ui, changePicker: null } })}
          />
          <motion.div
            key="pick-panel"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ type: 'spring', damping: 26, stiffness: 320 }}
            className="fixed inset-0 z-[71] flex items-center justify-center p-4 pointer-events-none"
          >
            <div
              className="pointer-events-auto w-full max-w-md overflow-hidden rounded-2xl border border-cyan-500/20 bg-slate-950/95 shadow-[0_0_60px_rgba(59,130,246,0.2)] backdrop-blur-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <h2 className="font-display text-sm font-bold uppercase tracking-widest text-cyan-200/90">
                  {title}
                </h2>
                <button
                  type="button"
                  className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-cyan-400"
                  onClick={() => onPatch({ ui: { ...ui, changePicker: null } })}
                  aria-label="Close"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <ul className="max-h-[50vh] overflow-y-auto p-2 font-data text-sm">
                {items.map((item) => (
                  <li key={item}>
                    <button
                      type="button"
                      className="w-full rounded-xl px-4 py-3 text-left text-slate-300 transition hover:bg-cyan-500/10 hover:text-cyan-200"
                      onClick={() => select(item)}
                    >
                      {item}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
