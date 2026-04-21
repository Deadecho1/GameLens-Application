/**
 * GameLens — central data bridge (single source of truth)
 * ---------------------------------------------------------------------------
 * BACKEND TEAM: Treat this object shape as the contract. The UI only reads
 * and writes these fields. Replace mock flows in App.jsx with API calls that
 * PATCH the same structure (or merge server payloads into this state).
 *
 * Suggested integration points:
 * - setup.*           → GET/POST games & versions; sync after catalog changes.
 * - processing.*     → WebSocket/SSE for logs; POST run/stop; folder scan → videoFiles.
 * - dashboard.*      → GET aggregated analytics for ANALYTICS tab.
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
   * ui — Frontend-only. Main navigation: activeMainTab drives SETUP | PROCESS | ANALYTICS.
   */
  ui: {
    /** Primary console tab: 'setup' | 'process' | 'analytics' */
    activeMainTab: 'setup',
    /**
     * “Change” selection card → opens list picker. null | 'game' | 'version'
     * BACKEND: not used; replace with API-driven picker if needed.
     */
    changePicker: null,
    /** Register new mission (game) modal */
    addGameModalOpen: false,
    /** Register new build (version) modal */
    addVersionModalOpen: false,
    newGameNameDraft: '',
    newVersionNameDraft: '',
  },

  /**
   * setup — Game / version catalog and current selection (SETUP tab + PROCESS context).
   */
  setup: {
    games: ['Elden Ring', 'Hades', 'Cyberpunk 2077'],
    versions: ['v1.0.1-stable', 'v1.1.0-alpha'],
    selectedGame: 'Elden Ring',
    selectedVersion: 'v1.1.0-alpha',
  },

  /**
   * processing — Pipeline state (PROCESS tab). status drives Run/Stop UI and optional badges.
   */
  processing: {
    pipelinePath: 'C:/Games/Captures/EldenRing',
    videoFiles: ['boss_fight_01.mp4', 'intro_cutscene.mp4'],
    selectedOption: 'only event',
    status: 'idle',
    logs: ['[INFO] System ready...', '[DEBUG] Waiting for user to click Run...'],
  },

  /**
   * dashboard — ANALYTICS tab (bento: bosses, items, runs, headline stats).
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

export function cloneInitialData() {
  return structuredClone(initialData);
}
