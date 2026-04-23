import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import { cloneInitialData, MOCK_VIDEOS_FOR_PATH, initialData } from './dataStore';
import Header from './components/Header';
import MainTabNav from './components/MainTabNav';
import ChangePickerModal from './components/ChangePickerModal';
import AddItemModal from './components/AddItemModal';
import MissionSuccessOverlay from './components/MissionSuccessOverlay';
import WorkflowTab from './components/tabs/WorkflowTab';
import AnalyticsTab from './components/tabs/AnalyticsTab';

/**
 * GameLens — MISSION START (workflow) + ANALYTICS. State mirrors dataStore.js.
 */

function App() {
  const [data, setData] = useState(() => cloneInitialData());
  const mockRunTimerRef = useRef(null);
  const prevStatusRef = useRef(initialData.processing.status);

  const mergePatch = useCallback((patch) => {
    setData((prev) => ({
      ...prev,
      ...(patch.ui ? { ui: { ...prev.ui, ...patch.ui } } : {}),
      ...(patch.setup ? { setup: { ...prev.setup, ...patch.setup } } : {}),
      ...(patch.processing ? { processing: { ...prev.processing, ...patch.processing } } : {}),
      ...(patch.dashboard ? { dashboard: { ...prev.dashboard, ...patch.dashboard } } : {}),
    }));
  }, []);

  useEffect(() => {
    return () => clearTimeout(mockRunTimerRef.current);
  }, []);

  /** When processing.status becomes completed, transition to ANALYTICS + success HUD. */
  useEffect(() => {
    const prev = prevStatusRef.current;
    const next = data.processing.status;
    if (prev !== 'completed' && next === 'completed') {
      setData((p) => ({
        ...p,
        ui: {
          ...p.ui,
          activeMainTab: 'analytics',
          completionCelebrationActive: true,
        },
      }));
    }
    prevStatusRef.current = next;
  }, [data.processing.status]);

  useEffect(() => {
    if (!data.ui.completionCelebrationActive) return undefined;
    const t = window.setTimeout(() => {
      setData((p) => ({
        ...p,
        ui: { ...p.ui, completionCelebrationActive: false },
      }));
    }, 3200);
    return () => window.clearTimeout(t);
  }, [data.ui.completionCelebrationActive]);

  const handleChooseFolder = () => {
    const paths = Object.keys(MOCK_VIDEOS_FOR_PATH);
    setData((prev) => {
      const idx = paths.indexOf(prev.processing.pipelinePath);
      const nextPath = paths[(idx + 1) % paths.length];
      return {
        ...prev,
        processing: {
          ...prev.processing,
          pipelinePath: nextPath,
          videoFiles: [...(MOCK_VIDEOS_FOR_PATH[nextPath] || [])],
        },
      };
    });
  };

  const handleRun = () => {
    clearTimeout(mockRunTimerRef.current);
    setData((prev) => ({
      ...prev,
      processing: {
        ...prev.processing,
        status: 'running',
        logs: [
          ...prev.processing.logs,
          `[RUN] Started (${prev.processing.selectedOption})…`,
        ],
      },
    }));
    mockRunTimerRef.current = setTimeout(() => {
      mockRunTimerRef.current = null;
      setData((prev) => ({
        ...prev,
        processing: {
          ...prev.processing,
          status: 'completed',
          logs: [...prev.processing.logs, '[RUN] Pipeline completed.'],
        },
      }));
    }, 4000);
  };

  const handleStop = () => {
    clearTimeout(mockRunTimerRef.current);
    mockRunTimerRef.current = null;
    setData((prev) => ({
      ...prev,
      processing: {
        ...prev.processing,
        status: 'stopped',
        logs: [...prev.processing.logs, '[WARN] Stopped by user.'],
      },
    }));
  };

  const handleClearLogs = () => {
    setData((prev) => ({
      ...prev,
      processing: { ...prev.processing, logs: [] },
    }));
  };

  const confirmAddGame = () => {
    setData((prev) => {
      const name = prev.ui.newGameNameDraft.trim();
      if (!name) return prev;
      const games = prev.setup.games.includes(name)
        ? prev.setup.games
        : [...prev.setup.games, name];
      return {
        ...prev,
        setup: { ...prev.setup, games, selectedGame: name },
        ui: { ...prev.ui, addGameModalOpen: false, newGameNameDraft: '' },
      };
    });
  };

  const confirmAddVersion = () => {
    setData((prev) => {
      const name = prev.ui.newVersionNameDraft.trim();
      if (!name) return prev;
      const versions = prev.setup.versions.includes(name)
        ? prev.setup.versions
        : [...prev.setup.versions, name];
      return {
        ...prev,
        setup: { ...prev.setup, versions, selectedVersion: name },
        ui: { ...prev.ui, addVersionModalOpen: false, newVersionNameDraft: '' },
      };
    });
  };

  const tab = data.ui.activeMainTab;

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-slate-950 text-slate-100">
      <div className="pointer-events-none fixed inset-0 gl-cyber-grid" aria-hidden />
      <div
        className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_120%_80%_at_50%_-20%,rgba(30,58,138,0.14),transparent)]"
        aria-hidden
      />
      <div className="gl-app-scanlines" aria-hidden />

      <MissionSuccessOverlay active={data.ui.completionCelebrationActive} />

      <div className="relative z-10">
        <Header data={data} />
        <MainTabNav data={data} onPatch={mergePatch} />

        <div className="relative min-h-[calc(100vh-8rem)]">
          <AnimatePresence mode="wait">
            {tab === 'workflow' && (
              <WorkflowTab
                key="workflow"
                data={data}
                onPatch={mergePatch}
                onAddGame={() =>
                  setData((p) => ({
                    ...p,
                    ui: { ...p.ui, addGameModalOpen: true, newGameNameDraft: '' },
                  }))
                }
                onAddVersion={() =>
                  setData((p) => ({
                    ...p,
                    ui: { ...p.ui, addVersionModalOpen: true, newVersionNameDraft: '' },
                  }))
                }
                onChooseFolder={handleChooseFolder}
                onRun={handleRun}
                onStop={handleStop}
                onClearLogs={handleClearLogs}
              />
            )}
            {tab === 'analytics' && (
              <AnalyticsTab key="analytics" data={data} onPatch={mergePatch} />
            )}
          </AnimatePresence>
        </div>

        <ChangePickerModal data={data} onPatch={mergePatch} />

        <AddItemModal
          open={data.ui.addGameModalOpen}
          title="Register game"
          draftValue={data.ui.newGameNameDraft}
          onDraftChange={(v) =>
            setData((prev) => ({ ...prev, ui: { ...prev.ui, newGameNameDraft: v } }))
          }
          onClose={() =>
            setData((prev) => ({ ...prev, ui: { ...prev.ui, addGameModalOpen: false } }))
          }
          onConfirm={confirmAddGame}
          inputId="add-game"
        />
        <AddItemModal
          open={data.ui.addVersionModalOpen}
          title="Register Version"
          draftValue={data.ui.newVersionNameDraft}
          onDraftChange={(v) =>
            setData((prev) => ({ ...prev, ui: { ...prev.ui, newVersionNameDraft: v } }))
          }
          onClose={() =>
            setData((prev) => ({ ...prev, ui: { ...prev.ui, addVersionModalOpen: false } }))
          }
          onConfirm={confirmAddVersion}
          inputId="add-version"
        />
      </div>
    </div>
  );
}

export default App;
