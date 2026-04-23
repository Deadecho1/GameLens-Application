/**
 * GameLens — central data bridge (single source of truth)
 * ---------------------------------------------------------------------------
 * BACKEND TEAM: Treat this object shape as the contract.
 *
 * - ui.activeMainTab     → 'workflow' | 'analytics'
 * - ui.workflowStep      → 1 Configure | 2 Initialize | 3 Execute (MISSION START tab only)
 * - ui.completionCelebrationActive → brief success HUD after processing.status → completed
 */

export const MOCK_VIDEOS_FOR_PATH = {
  'C:/Games/Captures/EldenRing': ['boss_fight_01.mp4', 'intro_cutscene.mp4', 'limgrave_02.mp4'],
  'C:/Games/Captures/Hades': ['run_17.mp4', 'escape_attempt.mp4'],
  'D:/Captures/GameLens': ['clip_alpha.mp4', 'clip_beta.mp4', 'full_session.mp4'],
};

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
    games: ['Elden Ring', 'Hades', 'Cyberpunk 2077'],
    versions: ['v1.0.1-stable', 'v1.1.0-alpha'],
    selectedGame: 'Elden Ring',
    selectedVersion: 'v1.1.0-alpha',
  },

  processing: {
    pipelinePath: 'C:/Games/Captures/EldenRing',
    videoFiles: ['boss_fight_01.mp4', 'intro_cutscene.mp4'],
    selectedOption: 'only event',
    status: 'idle',
    logs: ['[INFO] System ready...', '[DEBUG] Waiting for user to click Run...'],
  },

  /**
   * Analytics payloads.
   * GENERAL tab derives KPIs from runsHistory, bosses, and items only.
   * BOSSES global intel merges dashboard.bosses with per-session bossEncounters on runsHistory.
   */
  dashboard: {
    items: [
      { id: 1, name: 'Health Potion', popularity: 85, impact: 'High' },
      { id: 2, name: 'Iron Sword', popularity: 40, impact: 'Medium' },
      { id: 3, name: 'Shield', popularity: 62, impact: 'Medium' },
    ],
    /**
     * Boss catalog. globalLifespanSamples: optional extra detections (same time format as lifespan)
     * merged with all matching bossEncounters across runsHistory for aggregate BOSSES analytics.
     */
    bosses: [
      {
        id: 1,
        name: 'Malenia',
        lifespan: '05:20',
        status: 'Defeated',
        globalLifespanSamples: ['05:08', '06:01'],
      },
      {
        id: 2,
        name: 'Radahn',
        lifespan: '02:15',
        status: 'Alive',
        globalLifespanSamples: ['00:45', '01:12'],
      },
    ],
    /**
     * Analyzed sessions. bossEncounters: boss fights detected in that session (bossId + lifespan).
     */
    runsHistory: [
      {
        id: 'RUN-001',
        date: '2024-05-18',
        duration: '00:28:00',
        bossEncounters: [
          { bossId: 1, lifespan: '04:48' },
          { bossId: 2, lifespan: '01:52' },
        ],
      },
      {
        id: 'RUN-002',
        date: '2024-05-19',
        duration: '00:45:30',
        bossEncounters: [{ bossId: 1, lifespan: '05:55' }],
      },
      {
        id: 'RUN-003',
        date: '2024-05-20',
        duration: '00:30:00',
        bossEncounters: [
          { bossId: 1, lifespan: '05:20' },
          { bossId: 2, lifespan: '02:15' },
        ],
      },
      {
        id: 'RUN-004',
        date: '2024-05-20',
        duration: '00:52:10',
        bossEncounters: [{ bossId: 2, lifespan: '02:40' }],
      },
      {
        id: 'RUN-005',
        date: '2024-05-21',
        duration: '01:10:00',
        bossEncounters: [
          { bossId: 1, lifespan: '06:02' },
          { bossId: 2, lifespan: '01:58' },
        ],
      },
      {
        id: 'RUN-006',
        date: '2024-05-22',
        duration: '00:38:45',
        bossEncounters: [
          { bossId: 1, lifespan: '04:31' },
          { bossId: 2, lifespan: '02:22' },
        ],
      },
    ],
  },
};

export function cloneInitialData() {
  return structuredClone(initialData);
}
