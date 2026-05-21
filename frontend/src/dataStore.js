/**
 * GameLens — central data bridge (single source of truth)
 * ---------------------------------------------------------------------------
 * BACKEND TEAM: Treat this object shape as the contract.
 *
 * - ui.activeMainTab     → 'workflow' | 'analytics'
 * - ui.workflowStep      → 1 Configure | 2 Initialize | 3 Execute (MISSION START tab only)
 * - ui.completionCelebrationActive → brief success HUD after processing.status → completed
 */

export const initialData = {
  ui: {
    /** Main nav: unified mission flow vs analytics deck */
    activeMainTab: 'workflow',
    /**
     * Workflow stepper position (MISSION START tab).
     * 1 = Configure (game/version) · 2 = Initialize (upload/options) · 3 = Execute (run + terminal)
     */
    workflowStep: 1,
    /** Brief full-screen success pulse + banner when run completes (Framer-driven in App) */
    completionCelebrationActive: false,
    changePicker: null,
    addGameModalOpen: false,
    addVersionModalOpen: false,
    newGameNameDraft: '',
    newVersionNameDraft: '',
    /**
     * Analytics deck sub-view: 'general' | 'bosses' | 'items'
     * BACKEND: optional; omit when syncing API payloads.
     */
    analyticsSubTab: 'general',
  },

  setup: {
    games: [],
    versions: [],
    selectedGame: '',
    selectedVersion: '',
    user: {
      firstName: '',
      lastName: '',
      email: '',
      password: '',
      openAiKey: '',
    },
    fineTunedModels: [],   // [{name, modelId, dirName}] — populated from models/finetuned/
    selectedModel: 'base', // 'base' or a dirName from fineTunedModels
  },

  processing: {
    pipelinePath: '',
    videoFiles: [],
    selectedOption: 'only event',
    selectedModel: 'base',
    status: 'idle',
    logs: ['[INFO] System ready...'],
  },

  tuning: {
    status: 'idle',   // 'idle' | 'running' | 'completed' | 'stopped'
    logs: [],
  },

  /**
   * Analytics payloads.
   * GENERAL tab derives KPIs from runsHistory, bosses, and items only.
   * BOSSES global intel merges dashboard.bosses with per-session bossEncounters on runsHistory.
   */
  dashboard: {
    gameLibrary: {},
    items: [],
    bosses: [],
    runsHistory: [],
  },

  auth: {
    loggedIn: false,
    email: null,
    userId: null,
    syncStatus: 'idle',
    syncMessage: '',
  },
};

export function cloneInitialData() {
  return structuredClone(initialData);
}
