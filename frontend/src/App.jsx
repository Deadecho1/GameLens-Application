import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import { cloneInitialData } from "./dataStore";
import { IS_MOCK, getItems, getBosses, getRuns } from "./api/client";
import Header from "./components/Header";
import MainTabNav from "./components/MainTabNav";
import ChangePickerModal from "./components/ChangePickerModal";
import AddItemModal from "./components/AddItemModal";
import MissionSuccessOverlay from "./components/MissionSuccessOverlay";
import PostProcessingReviewModal from "./components/PostProcessingReviewModal";
import SettingsSidebar from "./components/SettingsSidebar";
import {
  extractPendingRunFromState,
  confirmPendingRunToLibrary,
  discardPendingRun,
  closeReviewWithoutSync,
} from "./utils/postProcessingRun";
import WorkflowTab from "./components/tabs/WorkflowTab";
import AnalyticsTab from "./components/tabs/AnalyticsTab";
import RunSessionAnalytics from "./components/analytics/runSession/RunSessionAnalytics";
import TuningTab from "./components/tabs/TuningTab";

/**
 * GameLens frontend orchestrator.
 * Primary source of truth is Qt over Electron IPC.
 */
function App() {
  const [data, setData] = useState(() => cloneInitialData());
  const [error, setError] = useState("");
  const [modalError, setModalError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const pollRef = useRef(null);
  const runsSnapshotRef = useRef([]);
  const reviewOpenRef = useRef(false);
  /** True on launch so stale `completed` from a prior session does not auto-open review. */
  const completionReviewDismissedRef = useRef(true);
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
        const reviewOpen =
          prev.ui.postProcessingReviewOpen || reviewOpenRef.current;
        const lastProcessedRun =
          prev.processing?.lastProcessedRun ??
          state.processing?.lastProcessedRun ??
          null;
        if (!reviewOpen) {
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
              postProcessingReviewOpen: false,
            },
            processing: {
              ...state.processing,
              lastProcessedRun,
              pendingRun: null,
              status:
                completionReviewDismissedRef.current &&
                state.processing?.status === "completed"
                  ? "idle"
                  : state.processing?.status,
            },
          };
        }
        const pendingRun =
          prev.processing?.pendingRun ??
          prev.processing?.lastProcessedRun ??
          extractPendingRunFromState(state, runsSnapshotRef.current);
        return {
          ...state,
          ui: {
            ...state.ui,
            postProcessingReviewOpen: true,
            activeMainTab:
              prev.ui.activeMainTab === "analytics"
                ? "workflow"
                : prev.ui.activeMainTab,
            completionCelebrationActive: false,
          },
          processing: {
            ...state.processing,
            status: "completed",
            pendingRun,
            lastProcessedRun: pendingRun ?? lastProcessedRun,
          },
        };
      });
      if (forceStartupAnalytics && !reviewOpenRef.current) {
        ipcRequest("ui:patch", {
          activeMainTab: "analytics",
          postProcessingReviewOpen: false,
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
        postProcessingReviewOpen: false,
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
    reviewOpenRef.current = Boolean(data.ui.postProcessingReviewOpen);
  }, [data.ui.postProcessingReviewOpen]);

  useEffect(() => {
    if (data.processing.status === "running") {
      completionReviewDismissedRef.current = false;
    }
    if (data.processing.status !== "completed") {
      runsSnapshotRef.current = data.dashboard?.runsHistory ?? [];
    }
  }, [data.processing.status, data.dashboard?.runsHistory]);

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
          if (patch.ui.postProcessingReviewOpen === false) {
            reviewOpenRef.current = false;
          }
          latestState = await ipcRequest("ui:patch", patch.ui);
        }

        if (patch.processing) {
          setData((prev) => ({
            ...prev,
            processing: {
              ...prev.processing,
              ...patch.processing,
              lastProcessedRun:
                patch.processing.lastProcessedRun !== undefined
                  ? patch.processing.lastProcessedRun
                  : prev.processing?.lastProcessedRun,
            },
          }));
        }

        if (latestState) {
          setData((prev) => ({
            ...latestState,
            processing: {
              ...latestState.processing,
              lastProcessedRun:
                prev.processing?.lastProcessedRun ??
                latestState.processing?.lastProcessedRun ??
                null,
              pendingRun: prev.processing?.pendingRun ?? null,
              status:
                completionReviewDismissedRef.current &&
                latestState.processing?.status === "completed"
                  ? "idle"
                  : latestState.processing?.status,
            },
          }));
          return;
        }

        setData((prev) => ({
          ...prev,
          ...(patch.ui ? { ui: { ...prev.ui, ...patch.ui } } : {}),
          ...(patch.setup ? { setup: { ...prev.setup, ...patch.setup } } : {}),
          ...(patch.processing
            ? {
                processing: {
                  ...prev.processing,
                  ...patch.processing,
                  lastProcessedRun:
                    patch.processing.lastProcessedRun !== undefined
                      ? patch.processing.lastProcessedRun
                      : prev.processing?.lastProcessedRun,
                },
              }
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
    try {
      const state = await ipcRequest("processing:run");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

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
      try {
        const state = await ipcRequest("auth:login", { email });
        setData(state);
      } catch (e) {
        setModalError(String(e?.message || e));
      }
    },
    [ipcRequest],
  );

  const handleLogout = useCallback(async () => {
    try {
      const state = await ipcRequest("auth:logout");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const handleSyncNow = useCallback(async () => {
    try {
      const state = await ipcRequest("auth:sync");
      setData(state);
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [ipcRequest]);

  const refreshDashboard = useCallback(async () => {
    const gameName = data.setup.selectedGame;
    const versionName = data.setup.selectedVersion;
    if (!gameName) return;
    try {
      let items;
      let bosses;
      let runsHistory;
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
      setData((prev) => ({
        ...prev,
        dashboard: { ...prev.dashboard, items, bosses, runsHistory },
      }));
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [
    data.setup.selectedGame,
    data.setup.selectedVersion,
    ipcRequest,
  ]);

  const resolveReviewRun = useCallback(
    (sourceData) => {
      const baseline = runsSnapshotRef.current ?? [];
      return (
        sourceData.processing?.pendingRun ??
        sourceData.processing?.lastProcessedRun ??
        extractPendingRunFromState(sourceData, baseline)
      );
    },
    [],
  );

  const finishReviewLocally = useCallback((nextData) => {
    reviewOpenRef.current = false;
    completionReviewDismissedRef.current = true;
    setData(nextData);
    ipcRequest("ui:patch", { postProcessingReviewOpen: false }).catch(() => {});
  }, [ipcRequest]);

  const handleReviewClose = useCallback(() => {
    const pending = resolveReviewRun(data);
    const baseline = runsSnapshotRef.current ?? [];
    finishReviewLocally(closeReviewWithoutSync(data, pending, baseline));
  }, [data, finishReviewLocally, resolveReviewRun]);

  const handleReviewDiscard = useCallback(() => {
    const pending = resolveReviewRun(data);
    const baseline = runsSnapshotRef.current ?? [];
    finishReviewLocally(discardPendingRun(data, pending, baseline));
  }, [data, finishReviewLocally, resolveReviewRun]);

  const handleReviewConfirm = useCallback(async () => {
    const pending = resolveReviewRun(data);
    const baseline = runsSnapshotRef.current ?? [];
    reviewOpenRef.current = false;
    completionReviewDismissedRef.current = true;
    setData(confirmPendingRunToLibrary(data, pending, baseline));
    ipcRequest("ui:patch", { postProcessingReviewOpen: false }).catch(() => {});
    try {
      await refreshDashboard();
    } catch (e) {
      setModalError(String(e?.message || e));
    }
  }, [data, ipcRequest, refreshDashboard, resolveReviewRun]);

  const handleReviewTuning = useCallback(() => {
    const pending = resolveReviewRun(data);
    const baseline = runsSnapshotRef.current ?? [];
    const next = discardPendingRun(data, pending, baseline);
    reviewOpenRef.current = false;
    completionReviewDismissedRef.current = true;
    setData({
      ...next,
      ui: { ...next.ui, activeMainTab: "tuning" },
    });
    ipcRequest("ui:patch", {
      postProcessingReviewOpen: false,
      activeMainTab: "tuning",
    }).catch(() => {});
  }, [data, ipcRequest, resolveReviewRun]);

  const handleReviewLastRun = useCallback(() => {
    const run = data.processing?.lastProcessedRun;
    if (!run) return;
    reviewOpenRef.current = true;
    setData((prev) => ({
      ...prev,
      ui: {
        ...prev.ui,
        activeMainTab: "workflow",
        postProcessingReviewOpen: true,
      },
      processing: {
        ...prev.processing,
        status: "completed",
        pendingRun: run,
      },
    }));
    ipcRequest("ui:patch", {
      activeMainTab: "workflow",
      postProcessingReviewOpen: true,
    }).catch(() => {});
  }, [data.processing?.lastProcessedRun, ipcRequest]);

  const tab = data.ui.activeMainTab;
  const pendingRun = data.ui.postProcessingReviewOpen
    ? resolveReviewRun(data)
    : null;

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

      <PostProcessingReviewModal
        open={Boolean(data.ui.postProcessingReviewOpen)}
        data={data}
        pendingRun={pendingRun}
        onClose={handleReviewClose}
        onDiscard={handleReviewDiscard}
        onConfirm={handleReviewConfirm}
        onGoToTuning={handleReviewTuning}
      />

      <div className="relative z-10">
        <Header
          data={data}
          onLogin={handleLogin}
          onLogout={handleLogout}
          onSyncNow={handleSyncNow}
          onOpenSettings={() => setSettingsOpen(true)}
        />
        <MainTabNav data={data} onPatch={mergePatch} />

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
                onReviewLastRun={handleReviewLastRun}
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
                      <p className="font-data text-sm text-slate-200 break-words">
                        {modalError}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setModalError("")}
                      className="shrink-0 rounded p-1 text-slate-400 hover:text-slate-200 transition-colors"
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
  );
}

export default App;
