import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence } from "framer-motion";
import { cloneInitialData } from "./dataStore";
import Header from "./components/Header";
import MainTabNav from "./components/MainTabNav";
import ChangePickerModal from "./components/ChangePickerModal";
import AddItemModal from "./components/AddItemModal";
import MissionSuccessOverlay from "./components/MissionSuccessOverlay";
import WorkflowTab from "./components/tabs/WorkflowTab";
import AnalyticsTab from "./components/tabs/AnalyticsTab";

/**
 * GameLens frontend orchestrator.
 * Primary source of truth is Qt over Electron IPC.
 */
function App() {
  const [data, setData] = useState(() => cloneInitialData());
  const [error, setError] = useState("");
  const pollRef = useRef(null);
  const prevStatusRef = useRef("idle");

  const ipcRequest = useCallback(async (method, params = {}) => {
    if (!window.gamelens?.request) {
      throw new Error(
        "Electron preload API not found. Run with `npm run electron:dev`.",
      );
    }
    return window.gamelens.request(method, params);
  }, []);

  const refreshState = useCallback(async () => {
    try {
      const state = await ipcRequest("state:get");
      setData(state);
      setError("");
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [ipcRequest]);

  useEffect(() => {
    refreshState();
    pollRef.current = window.setInterval(refreshState, 1500);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [refreshState]);

  /** On completed edge, trigger brief overlay exactly like the mock app behavior. */
  useEffect(() => {
    const prev = prevStatusRef.current;
    const next = data.processing.status;
    if (prev !== "completed" && next === "completed") {
      setData((p) => ({
        ...p,
        ui: {
          ...p.ui,
          activeMainTab: "analytics",
          completionCelebrationActive: true,
        },
      }));
      ipcRequest("ui:patch", {
        activeMainTab: "analytics",
        completionCelebrationActive: true,
      }).catch(() => {});
    }
    prevStatusRef.current = next;
  }, [data.processing.status, ipcRequest]);

  useEffect(() => {
    if (!data.ui.completionCelebrationActive) return undefined;
    const t = window.setTimeout(() => {
      setData((p) => ({
        ...p,
        ui: { ...p.ui, completionCelebrationActive: false },
      }));
      ipcRequest("ui:patch", { completionCelebrationActive: false }).catch(
        () => {},
      );
    }, 3200);
    return () => window.clearTimeout(t);
  }, [data.ui.completionCelebrationActive, ipcRequest]);

  const mergePatch = useCallback(
    async (patch) => {
      try {
        if (patch.ui) {
          const state = await ipcRequest("ui:patch", patch.ui);
          setData(state);
          return;
        }

        if (patch.setup?.selectedGame) {
          const state = await ipcRequest("setup:select_game", {
            game: patch.setup.selectedGame,
          });
          setData(state);
          return;
        }

        if (patch.setup?.selectedVersion) {
          const state = await ipcRequest("setup:select_version", {
            version: patch.setup.selectedVersion,
          });
          setData(state);
          return;
        }

        if (patch.processing?.selectedOption) {
          const state = await ipcRequest("processing:set_option", {
            option: patch.processing.selectedOption,
          });
          setData(state);
          return;
        }

        setData((prev) => ({
          ...prev,
          ...(patch.ui ? { ui: { ...prev.ui, ...patch.ui } } : {}),
          ...(patch.setup ? { setup: { ...prev.setup, ...patch.setup } } : {}),
          ...(patch.processing
            ? { processing: { ...prev.processing, ...patch.processing } }
            : {}),
          ...(patch.dashboard
            ? { dashboard: { ...prev.dashboard, ...patch.dashboard } }
            : {}),
        }));
      } catch (e) {
        setError(String(e?.message || e));
      }
    },
    [ipcRequest],
  );

  const handleChooseFolder = useCallback(async () => {
    try {
      const folder = await window.gamelens?.chooseFolder?.();
      if (!folder) return;
      const state = await ipcRequest("processing:stage_folder", {
        pipeline_path: folder,
      });
      setData(state);
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const handleRun = useCallback(async () => {
    try {
      const state = await ipcRequest("processing:run");
      setData(state);
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const handleStop = useCallback(async () => {
    try {
      const state = await ipcRequest("processing:stop");
      setData(state);
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const handleClearLogs = useCallback(async () => {
    try {
      const state = await ipcRequest("processing:clear_logs");
      setData(state);
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const confirmAddGame = useCallback(async () => {
    try {
      const name = data.ui.newGameNameDraft.trim();
      if (!name) return;
      const state = await ipcRequest("setup:add_game", { name });
      setData((prev) => ({
        ...state,
        ui: { ...state.ui, addGameModalOpen: false, newGameNameDraft: "" },
      }));
      await ipcRequest("ui:patch", {
        addGameModalOpen: false,
        newGameNameDraft: "",
      });
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [data.ui.newGameNameDraft, ipcRequest]);

  const confirmAddVersion = useCallback(async () => {
    try {
      const name = data.ui.newVersionNameDraft.trim();
      if (!name) return;
      const state = await ipcRequest("setup:add_version", {
        game_name: data.setup.selectedGame,
        version_name: name,
      });
      setData((prev) => ({
        ...state,
        ui: {
          ...state.ui,
          addVersionModalOpen: false,
          newVersionNameDraft: "",
        },
      }));
      await ipcRequest("ui:patch", {
        addVersionModalOpen: false,
        newVersionNameDraft: "",
      });
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [data.setup.selectedGame, data.ui.newVersionNameDraft, ipcRequest]);

  const tab = data.ui.activeMainTab;

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-slate-950 text-slate-100">
      <div
        className="pointer-events-none fixed inset-0 gl-cyber-grid"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_120%_80%_at_50%_-20%,rgba(30,58,138,0.14),transparent)]"
        aria-hidden
      />
      <div className="gl-app-scanlines" aria-hidden />

      <MissionSuccessOverlay active={data.ui.completionCelebrationActive} />

      <div className="relative z-10">
        <Header data={data} />
        <MainTabNav data={data} onPatch={mergePatch} />

        {error ? (
          <div className="mx-auto mt-3 max-w-[1800px] rounded-lg border border-red-500/40 bg-red-950/40 px-4 py-2 font-data text-xs text-red-200">
            IPC error: {error}
          </div>
        ) : null}

        <div className="relative min-h-[calc(100vh-8rem)]">
          <AnimatePresence mode="wait">
            {tab === "workflow" && (
              <WorkflowTab
                key="workflow"
                data={data}
                onPatch={mergePatch}
                onAddGame={() =>
                  mergePatch({
                    ui: {
                      ...data.ui,
                      addGameModalOpen: true,
                      newGameNameDraft: "",
                    },
                  })
                }
                onAddVersion={() =>
                  mergePatch({
                    ui: {
                      ...data.ui,
                      addVersionModalOpen: true,
                      newVersionNameDraft: "",
                    },
                  })
                }
                onChooseFolder={handleChooseFolder}
                onRun={handleRun}
                onStop={handleStop}
                onClearLogs={handleClearLogs}
              />
            )}
            {tab === "analytics" && (
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
            setData((prev) => ({
              ...prev,
              ui: { ...prev.ui, newGameNameDraft: v },
            }))
          }
          onClose={() =>
            mergePatch({ ui: { ...data.ui, addGameModalOpen: false } })
          }
          onConfirm={confirmAddGame}
          inputId="add-game"
        />
        <AddItemModal
          open={data.ui.addVersionModalOpen}
          title="Register Version"
          draftValue={data.ui.newVersionNameDraft}
          onDraftChange={(v) =>
            setData((prev) => ({
              ...prev,
              ui: { ...prev.ui, newVersionNameDraft: v },
            }))
          }
          onClose={() =>
            mergePatch({ ui: { ...data.ui, addVersionModalOpen: false } })
          }
          onConfirm={confirmAddVersion}
          inputId="add-version"
        />
      </div>
    </div>
  );
}

export default App;
