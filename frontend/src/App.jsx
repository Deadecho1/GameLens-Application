import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import { cloneInitialData } from "./dataStore";
import { IS_MOCK, getItems, getBosses, getRuns } from "./api/client";
import Header from "./components/Header";
import MainTabNav from "./components/MainTabNav";
import ChangePickerModal from "./components/ChangePickerModal";
import AddItemModal from "./components/AddItemModal";
import SettingsSidebar from "./components/SettingsSidebar";
import WorkflowTab from "./components/tabs/WorkflowTab";
import AnalyticsTab from "./components/tabs/AnalyticsTab";
import RunSessionAnalytics from "./components/analytics/runSession/RunSessionAnalytics";
import TuningTab from "./components/tabs/TuningTab";
import WelcomeScreen from "./components/WelcomeScreen";
import TitleBar from "./components/TitleBar";
import {
  clearGuestModeContinued,
  persistGuestModeContinued,
  readGuestModeContinued,
} from "./utils/guestMode";
import { EXIT_UI_PATCH, mergeExitSessionState } from "./utils/sessionExit";

/**
 * GameLens frontend orchestrator.
 * Primary source of truth is Qt over Electron IPC.
 */
function App() {
  const [data, setData] = useState(() => cloneInitialData());
  const [error, setError] = useState("");
  const [modalError, setModalError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [guestContinued, setGuestContinued] = useState(readGuestModeContinued);
  const pollRef = useRef(null);
  /** First IPC sync forces Analytics so stale backend `workflow` tab cannot win on startup. */
  const startupAnalyticsAppliedRef = useRef(false);

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
      const forceStartupAnalytics = !startupAnalyticsAppliedRef.current;
      setData((prev) => {
        const activeMainTab = forceStartupAnalytics
          ? "analytics"
          : state.ui?.activeMainTab ?? "analytics";
        if (forceStartupAnalytics) {
          startupAnalyticsAppliedRef.current = true;
        }
        return {
          ...state,
          ui: {
            ...state.ui,
            activeMainTab,
          },
          processing: {
            ...state.processing,
            status:
              forceStartupAnalytics &&
              state.processing?.status === "completed"
                ? "idle"
                : state.processing?.status,
          },
        };
      });
      if (forceStartupAnalytics) {
        ipcRequest("ui:patch", {
          activeMainTab: "analytics",
        }).catch(() => {});
      }
      setError("");
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [ipcRequest]);

  /** Keep Analytics visible before the first state:get returns (IPC may still say workflow). */
  useEffect(() => {
    setData((prev) => ({
      ...prev,
      ui: {
        ...prev.ui,
        activeMainTab: "analytics",
      },
    }));
  }, []);

  useEffect(() => {
    refreshState();
    pollRef.current = window.setInterval(refreshState, 1500);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [refreshState]);

  useEffect(() => {
    if (data.ui.activeMainTab !== "analytics") return;
    const gameName = (data.setup.selectedGame ?? "").trim();
    const versionName = data.setup.selectedVersion;
    if (!gameName) return;

    let cancelled = false;

    (async () => {
      try {
        let items, bosses, runsHistory;
        if (IS_MOCK) {
          [items, bosses, runsHistory] = await Promise.all([
            getItems(null, gameName, versionName),
            getBosses(null, gameName, versionName),
            getRuns(null, gameName, versionName),
          ]);
        } else {
          [items, bosses, runsHistory] = await Promise.all([
            ipcRequest("dashboard:items", {
              game_name: gameName,
              version_name: versionName,
            }),
            ipcRequest("dashboard:bosses", {
              game_name: gameName,
              version_name: versionName,
            }),
            ipcRequest("dashboard:runs", {
              game_name: gameName,
              version_name: versionName,
            }),
          ]);
        }
        if (!cancelled) {
          setData((prev) => ({
            ...prev,
            dashboard: { ...prev.dashboard, items, bosses, runsHistory },
          }));
        }
      } catch (e) {
        if (!cancelled) setModalError(String(e?.message || e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    data.ui.activeMainTab,
    data.setup.selectedGame,
    data.setup.selectedVersion,
    ipcRequest,
  ]);

  const mergePatch = useCallback(
    async (patch) => {
      try {
        let latestState = null;

        if (patch.setup?.selectedGame) {
          latestState = await ipcRequest("setup:select_game", {
            game: patch.setup.selectedGame,
          });
        }

        if (patch.setup?.selectedVersion) {
          latestState = await ipcRequest("setup:select_version", {
            version: patch.setup.selectedVersion,
          });
        }

        if (patch.processing?.selectedModel !== undefined) {
          latestState = await ipcRequest("processing:set_model", {
            model: patch.processing.selectedModel,
          });
        }

        if (patch.processing?.selectedOption !== undefined) {
          latestState = await ipcRequest("processing:set_option", {
            option: patch.processing.selectedOption,
          });
        }

        const hasProcessingPatch =
          patch.processing && typeof patch.processing === "object";

        if (
          hasProcessingPatch &&
          Object.prototype.hasOwnProperty.call(
            patch.processing,
            "pipelinePath",
          ) &&
          typeof patch.processing.pipelinePath === "string"
        ) {
          latestState = await ipcRequest("processing:set_pipeline_path", {
            pipeline_path: patch.processing.pipelinePath,
          });
        }

        if (
          hasProcessingPatch &&
          Object.prototype.hasOwnProperty.call(
            patch.processing,
            "videoFiles",
          ) &&
          Array.isArray(patch.processing.videoFiles)
        ) {
          latestState = await ipcRequest("processing:stage_files", {
            file_names: patch.processing.videoFiles,
            file_paths: Array.isArray(patch.processing.videoFilePaths)
              ? patch.processing.videoFilePaths
              : [],
          });
        }

        if (patch.setup?.user?.openAiKey !== undefined) {
          latestState = await ipcRequest("setup:save_settings", {
            openAiKey: patch.setup.user.openAiKey,
          });
        }

        if (patch.ui) {
          latestState = await ipcRequest("ui:patch", patch.ui);
        }

        if (latestState) {
          setData(latestState);
          return;
        }

        setData((prev) => ({
          ...prev,
          ...(patch.ui ? { ui: { ...prev.ui, ...patch.ui } } : {}),
          ...(patch.setup ? { setup: { ...prev.setup, ...patch.setup } } : {}),
          ...(patch.auth ? { auth: { ...prev.auth, ...patch.auth } } : {}),
          ...(patch.processing
            ? { processing: { ...prev.processing, ...patch.processing } }
            : {}),
          ...(patch.dashboard
            ? { dashboard: { ...prev.dashboard, ...patch.dashboard } }
            : {}),
        }));
      } catch (e) {
        setModalError(String(e?.message || e));
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
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const handleRun = useCallback(async () => {
    if (!data.setup?.user?.openAiKey) {
      setModalError("OpenAI API key is not configured. Add it in Settings before running the pipeline.");
      return;
    }
    try {
      const state = await ipcRequest("processing:run");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest, data.setup?.user?.openAiKey]);

  const handleStop = useCallback(async () => {
    try {
      const state = await ipcRequest("processing:stop");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const handleClearLogs = useCallback(async () => {
    try {
      const state = await ipcRequest("processing:clear_logs");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const confirmAddGame = useCallback(async () => {
    try {
      const name = data.ui.newGameNameDraft.trim();
      if (!name) return;
      const state = await ipcRequest("setup:add_game", { name });
      setData({
        ...state,
        ui: { ...state.ui, addGameModalOpen: false, newGameNameDraft: "" },
      });
      await ipcRequest("ui:patch", {
        addGameModalOpen: false,
        newGameNameDraft: "",
      });
    } catch (e) {
      setModalError(String(e?.message || e));
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
      setData({
        ...state,
        ui: {
          ...state.ui,
          addVersionModalOpen: false,
          newVersionNameDraft: "",
        },
      });
      await ipcRequest("ui:patch", {
        addVersionModalOpen: false,
        newVersionNameDraft: "",
      });
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [data.setup.selectedGame, data.ui.newVersionNameDraft, ipcRequest]);

  const handleLogin = useCallback(
    async (email) => {
      const state = await ipcRequest("auth:login", { email });
      clearGuestModeContinued();
      setGuestContinued(false);
      setData((prev) => ({
        ...state,
        ui: {
          ...state.ui,
          activeMainTab: "analytics",
        },
      }));
      ipcRequest("ui:patch", {
        activeMainTab: "analytics",
      }).catch(() => {});
    },
    [ipcRequest],
  );

  const handleHeaderLogin = useCallback(
    async (email) => {
      try {
        await handleLogin(email);
      } catch (e) {
        setModalError(String(e?.message || e));
        throw e;
      }
    },
    [handleLogin],
  );

  const handleGuestContinue = useCallback(() => {
    persistGuestModeContinued();
    setGuestContinued(true);
    setData((prev) => ({
      ...prev,
      ui: {
        ...prev.ui,
        activeMainTab: "analytics",
      },
    }));
    ipcRequest("ui:patch", {
      activeMainTab: "analytics",
    }).catch(() => {});
  }, [ipcRequest]);

  const handleLogout = useCallback(async () => {
    const wasLoggedIn = Boolean(data.auth?.loggedIn);
    const wasRunning = data.processing?.status === "running";

    clearGuestModeContinued();
    setGuestContinued(false);
    setSettingsOpen(false);
    setModalError("");

    setData((prev) => mergeExitSessionState(prev));

    try {
      if (wasRunning) {
        try {
          await ipcRequest("processing:stop");
        } catch {
          /* still exit session */
        }
      }

      if (wasLoggedIn) {
        await ipcRequest("auth:logout");
      }

      await ipcRequest("ui:patch", EXIT_UI_PATCH);

      const state = await ipcRequest("state:get");
      setData((prev) => mergeExitSessionState(prev, state));
    } catch (e) {
      setModalError(String(e?.message || e));
      setData((prev) => mergeExitSessionState(prev));
    }
  }, [data.auth?.loggedIn, data.processing?.status, ipcRequest]);

  const handleUpdateEmail = useCallback(
    async (email) => {
      const trimmed = (email ?? "").trim();
      if (!trimmed) return;
      try {
        const state = await ipcRequest("auth:login", { email: trimmed });
        setData(state);
      } catch (e) {
        setModalError(String(e?.message || e));
        throw e;
      }
    },
    [ipcRequest],
  );

  const handleSyncNow = useCallback(async () => {
    try {
      const state = await ipcRequest("auth:sync");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const tab = data.ui.activeMainTab;

  const showWelcome = !data.auth?.loggedIn && !guestContinued;

  if (showWelcome) {
    return (
      <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-base text-slate-200">
        <TitleBar />
        <div className="flex min-h-0 flex-1 overflow-hidden">
          <div className="min-h-0 flex-1 overflow-auto">
            <WelcomeScreen
              onLogin={handleLogin}
              onGuestContinue={handleGuestContinue}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-base text-slate-200">
      <TitleBar />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-slate-950 text-base text-slate-200">
          <div
            className="pointer-events-none absolute inset-0 gl-cyber-grid"
            aria-hidden
          />
          <div
            className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_120%_80%_at_50%_-20%,rgba(30,58,138,0.14),transparent)]"
            aria-hidden
          />
          <div className="gl-app-scanlines pointer-events-none absolute inset-0" aria-hidden />

          <div className="relative z-10 flex min-h-0 flex-1 flex-col overflow-hidden">
            <Header
              data={data}
              onLogin={handleHeaderLogin}
              onLogout={handleLogout}
              onSyncNow={handleSyncNow}
              onOpenSettings={() => setSettingsOpen(true)}
            />
            <MainTabNav data={data} onPatch={mergePatch} />

            <div className="relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto pb-8">
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
            {tab === "runSession" && (
              <RunSessionAnalytics key="runSession" data={data} />
            )}
            {tab === "tuning" && (
              <TuningTab key="tuning" data={data} ipcRequest={ipcRequest} />
            )}
            </AnimatePresence>
            </div>

            <ChangePickerModal data={data} onPatch={mergePatch} />

            <AddItemModal
          open={data.ui.addGameModalOpen}
          title="Register game"
          draftValue={data.ui.newGameNameDraft}
          onDraftChange={(v) => {
            setData((prev) => ({
              ...prev,
              ui: { ...prev.ui, newGameNameDraft: v },
            }));
            mergePatch({ ui: { newGameNameDraft: v } });
          }}
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
          onDraftChange={(v) => {
            setData((prev) => ({
              ...prev,
              ui: { ...prev.ui, newVersionNameDraft: v },
            }));
            mergePatch({ ui: { newVersionNameDraft: v } });
          }}
          onClose={() =>
            mergePatch({ ui: { ...data.ui, addVersionModalOpen: false } })
          }
          onConfirm={confirmAddVersion}
              inputId="add-version"
            />
            <SettingsSidebar
              data={data}
              onPatch={mergePatch}
              onUpdateEmail={handleUpdateEmail}
              open={settingsOpen}
              onClose={() => setSettingsOpen(false)}
            />

            <AnimatePresence>
              {modalError && (
                <>
                  <motion.div
                key="err-backdrop"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[80] bg-slate-950/70 backdrop-blur-sm"
                onClick={() => setModalError("")}
              />
              <motion.div
                key="err-modal"
                role="alertdialog"
                aria-modal="true"
                initial={{ opacity: 0, scale: 0.94, y: 12 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.94, y: 8 }}
                transition={{ type: "spring", damping: 24, stiffness: 320 }}
                className="fixed inset-0 z-[81] flex items-center justify-center p-4 pointer-events-none"
              >
                <div className="pointer-events-auto w-full max-w-md rounded-xl border border-red-500/30 bg-slate-900 shadow-2xl">
                  <div className="flex items-start gap-3 p-5">
                    <AlertTriangle
                      className="mt-0.5 shrink-0 text-red-400"
                      size={20}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-display text-sm font-semibold uppercase tracking-widest text-red-300 mb-1">
                        Error
                      </p>
                      <p className="font-data text-base text-slate-200 break-words">
                        {modalError}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setModalError("")}
                      className="shrink-0 rounded p-1 text-slate-300 hover:text-slate-200 transition-colors"
                      aria-label="Dismiss"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
