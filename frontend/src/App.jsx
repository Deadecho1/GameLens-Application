import { useCallback, useEffect, useRef, useState } from 'react';
import { cloneInitialData, MOCK_VIDEOS_FOR_PATH } from './dataStore';
import Header from './components/Header';
import AddItemModal from './components/AddItemModal';
import ProcessingModal from './components/ProcessingModal';
import Dashboard from './components/Dashboard';

/**
 * GameLens SPA — single source of truth: `data` cloned from dataStore.initialData.
 *
 * BACKEND INTEGRATION (replace mock handlers):
 * - mergePatch: apply server-driven partial updates to the same shape.
 * - handleRun / handleStop: POST to job API; subscribe for logs + status.
 * - handleChooseFolder: return path from OS picker, then GET scan → processing.videoFiles.
 * - On job completion: set processing.status to 'completed' and PATCH dashboard.* from report API.
 */

function App() {
  const [data, setData] = useState(() => cloneInitialData());
  const mockRunTimerRef = useRef(null);

  /** Shallow-merge into top-level slices (ui, setup, processing, dashboard). */
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

  /** Mock: cycle pipelinePath through MOCK_VIDEOS_FOR_PATH keys and refresh videoFiles. */
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

  /** Mock run: sets running, then completes after 4s unless Stop clears the timer. */
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
        ui: { ...prev.ui, processingModalOpen: false },
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
    const name = data.ui.newGameNameDraft.trim();
    if (!name) return;
    setData((prev) => {
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
    const name = data.ui.newVersionNameDraft.trim();
    if (!name) return;
    setData((prev) => {
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

  const showPlaceholder = data.processing.status !== 'completed';

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 antialiased">
      <Header data={data} onPatch={mergePatch} />

      <AddItemModal
        open={data.ui.addGameModalOpen}
        title="Add new game"
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
        title="Add new version"
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

      <ProcessingModal
        data={data}
        onPatch={mergePatch}
        onRun={handleRun}
        onStop={handleStop}
        onClearLogs={handleClearLogs}
        onChooseFolder={handleChooseFolder}
      />

      <main>
        {showPlaceholder ? (
          <div className="mx-auto max-w-3xl px-4 py-16 text-center">
            <div className="rounded-2xl border border-dashed border-slate-800 bg-slate-800/20 px-8 py-20 transition hover:border-slate-700">
              <p className="text-lg text-slate-400">
                Select a game and version, then use <span className="text-blue-400">Process Clip</span> to
                run the pipeline.
              </p>
              <p className="mt-4 text-sm text-slate-600">
                When <code className="rounded bg-slate-950 px-1.5 py-0.5 text-slate-500">processing.status</code>{' '}
                is <code className="text-emerald-500/90">completed</code>, the dashboard appears here.
              </p>
            </div>
          </div>
        ) : (
          <Dashboard data={data} onPatch={mergePatch} />
        )}
      </main>
    </div>
  );
}

export default App;
