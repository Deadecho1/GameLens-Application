import { useCallback, useEffect, useState } from 'react';
import { Minus, Square, X } from 'lucide-react';

const dragStyle = { WebkitAppRegion: 'drag' };
const noDragStyle = { WebkitAppRegion: 'no-drag' };

function isElectron() {
  return Boolean(window.gamelens?.windowControls);
}

/**
 * Custom frameless window title bar (Electron). Hidden in pure browser dev.
 */
export default function TitleBar() {
  const [maximized, setMaximized] = useState(false);
  const electron = isElectron();

  const syncMaximized = useCallback(async () => {
    const controls = window.gamelens?.windowControls;
    if (!controls?.isMaximized) return;
    try {
      setMaximized(await controls.isMaximized());
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!electron) return undefined;
    syncMaximized();
    const controls = window.gamelens.windowControls;
    const off = controls.onMaximizeChange?.((isMax) => setMaximized(isMax));
    return () => {
      if (typeof off === 'function') off();
    };
  }, [electron, syncMaximized]);

  if (!electron) return null;

  const controls = window.gamelens.windowControls;

  return (
    <header
      className="relative z-[200] flex h-8 shrink-0 select-none items-stretch border-b border-slate-800/90 bg-slate-900"
      style={dragStyle}
    >
      <div
        className="flex min-w-0 flex-1 items-center gap-2 px-3"
        style={dragStyle}
      >
        <span
          className="font-display text-[9px] font-bold uppercase tracking-[0.28em] text-cyan-500/70"
          aria-hidden
        >
          GameLens
        </span>
      </div>

      <div className="flex items-stretch" style={noDragStyle}>
        <WindowControlButton
          label="Minimize"
          onClick={() => controls.minimize()}
          className="hover:bg-slate-800"
        >
          <Minus className="h-3.5 w-3.5" strokeWidth={2} aria-hidden />
        </WindowControlButton>
        <WindowControlButton
          label={maximized ? 'Restore' : 'Maximize'}
          onClick={async () => {
            controls.maximize();
            await syncMaximized();
          }}
          className="hover:bg-slate-800"
        >
          <Square className="h-3 w-3" strokeWidth={2} aria-hidden />
        </WindowControlButton>
        <WindowControlButton
          label="Close"
          onClick={() => controls.close()}
          className="hover:bg-red-600 hover:text-white"
        >
          <X className="h-3.5 w-3.5" strokeWidth={2} aria-hidden />
        </WindowControlButton>
      </div>
    </header>
  );
}

function WindowControlButton({ label, onClick, className = '', children }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      style={noDragStyle}
      className={`flex h-8 w-11 items-center justify-center text-slate-400 transition ${className}`}
    >
      {children}
    </button>
  );
}
