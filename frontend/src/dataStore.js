/**
 * GameLens — central data bridge (single source of truth)
 * ---------------------------------------------------------------------------
 * BACKEND TEAM: Treat this object shape as the contract. The UI only reads
 * and writes these fields. Replace mock flows in App.jsx with API calls that
 * PATCH the same structure (or merge server payloads into this state).
 *
 * Suggested integration points:
 * - setup.*           → GET/POST games & versions; sync dropdowns after save.
 * - processing.*     → WebSocket or SSE for logs; POST run/stop; GET folder scan for videoFiles.
 * - dashboard.*      → GET aggregated report when processing.status becomes 'completed'.
 * - ui.*             → Frontend-only (omit from API); safe to strip on the server.
 */

/** Mock video filenames — BACKEND: replace with real directory listing for processing.pipelinePath */
export const MOCK_VIDEOS_FOR_PATH = {
  'C:/Games/Captures/EldenRing': ['boss_fight_01.mp4', 'intro_cutscene.mp4', 'limgrave_02.mp4'],
  'C:/Games/Captures/Hades': ['run_17.mp4', 'escape_attempt.mp4'],
  'D:/Captures/GameLens': ['clip_alpha.mp4', 'clip_beta.mp4', 'full_session.mp4'],
};

export const initialData = {
  /**
   * ui — Frontend-only modal/tab/input state. Not persisted to backend.
   */
  ui: {
    /** Gear icon: mission catalog / add game & version (Config drawer) */
    configSidebarOpen: false,
    /** When true, the large “Process Clip” workflow modal is visible */
    processingModalOpen: false,
    /** “Add new game” centered modal */
    addGameModalOpen: false,
    /** “Add new version” centered modal */
    addVersionModalOpen: false,
    /** Dashboard tab key: 'summary' | 'combat' | 'inventory' */
    dashboardActiveTab: 'summary',
    /** Bound to the text field in Add Game modal — BACKEND: not used */
    newGameNameDraft: '',
    /** Bound to the text field in Add Version modal — BACKEND: not used */
    newVersionNameDraft: '',
  },

  /**
   * setup — Game / version selection shown in the header.
   * BACKEND: Populate games[] and versions[] from your catalog API.
   *          Push updates when user adds a game/version (POST then refresh lists).
   */
  setup: {
    games: ['Elden Ring', 'Hades', 'Cyberpunk 2077'],
    versions: ['v1.0.1-stable', 'v1.1.0-alpha'],
    selectedGame: 'Elden Ring',
    selectedVersion: 'v1.1.0-alpha',
  },

  /**
   * processing — Active clip pipeline. BACKEND owns authoritative values during a run.
   *
   * pipelinePath   — Root folder the backend should scan (user picks folder in UI).
   * videoFiles     — BACKEND: set to list of discovered video filenames after scan.
   * selectedOption — CLI mode: 'only event' | 'only export' | 'verbose'
   * status         — 'idle' | 'running' | 'stopped' | 'completed' — drives SPA layout.
   * logs           — BACKEND: append lines (stdout/stderr); UI appends only for local mock.
   */
  processing: {
    pipelinePath: 'C:/Games/Captures/EldenRing',
    videoFiles: ['boss_fight_01.mp4', 'intro_cutscene.mp4'],
    selectedOption: 'only event',
    status: 'idle',
    logs: ['[INFO] System ready...', '[DEBUG] Waiting for user to click Run...'],
  },

  /**
   * dashboard — Post-run analytics. BACKEND: fill when processing completes (or poll a job id).
   *
   * stats          — Summary cards (totals, durations).
   * items          — Popularity / impact for bar list or charts.
   * bosses         — Combat tab: Name, lifespan, Alive/Defeated etc.
   * runsHistory    — Runs tab list rows.
   */
  dashboard: {
    stats: {
      totalRuns: 12,
      averageRunTime: '00:42:15',
      longestRun: '01:15:30',
      totalItemsFound: 154,
    },
    items: [
      { id: 1, name: 'Health Potion', popularity: 85, impact: 'High' },
      { id: 2, name: 'Iron Sword', popularity: 40, impact: 'Medium' },
      { id: 3, name: 'Shield', popularity: 62, impact: 'Medium' },
    ],
    bosses: [
      { id: 1, name: 'Malenia', lifespan: '05:20', status: 'Defeated' },
      { id: 2, name: 'Radahn', lifespan: '02:15', status: 'Alive' },
    ],
    runsHistory: [
      { id: 'RUN-001', date: '2024-05-20', duration: '00:30:00' },
      { id: 'RUN-002', date: '2024-05-21', duration: '01:10:00' },
    ],
  },
};

/** Deep clone for React initial state (no functions in store) */
export function cloneInitialData() {
  return structuredClone(initialData);
}
