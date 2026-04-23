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
    /**
     * Item catalog. category: offensive | defensive | utility (Synergy Lab filters).
     * logicTag: short combat label for UI tooltips.
     */
    items: [
      { id: 1, name: 'Power Potion', popularity: 85, impact: 'High', category: 'utility', logicTag: 'Burst sustain' },
      { id: 2, name: 'Fire Sword', popularity: 40, impact: 'Medium', category: 'offensive', logicTag: 'High damage' },
      { id: 3, name: 'Tower Shield', popularity: 62, impact: 'Medium', category: 'defensive', logicTag: 'Block stability' },
      { id: 4, name: 'Frost Dagger', popularity: 55, impact: 'Medium', category: 'offensive', logicTag: 'Slow procs' },
      { id: 5, name: 'Arcane Focus', popularity: 48, impact: 'High', category: 'utility', logicTag: 'Spell amp' },
      { id: 6, name: 'Heavy Plate', popularity: 71, impact: 'High', category: 'defensive', logicTag: 'Damage soak' },
      { id: 7, name: 'Venom Flask', popularity: 33, impact: 'Medium', category: 'offensive', logicTag: 'DoT pressure' },
      { id: 8, name: 'Healing Draught', popularity: 90, impact: 'High', category: 'utility', logicTag: 'Recovery' },
      { id: 9, name: 'Thunder Mallet', popularity: 28, impact: 'Medium', category: 'offensive', logicTag: 'Stagger' },
      { id: 10, name: 'Void Charm', popularity: 52, impact: 'Low', category: 'utility', logicTag: 'Resource regen' },
      { id: 11, name: 'Buckler', popularity: 44, impact: 'Medium', category: 'defensive', logicTag: 'Parry window' },
      { id: 12, name: 'Explosive Seed', popularity: 22, impact: 'Low', category: 'offensive', logicTag: 'AoE chip' },
    ],
    /**
     * Boss catalog. globalLifespanSamples: merged with bossEncounters for lifespan stats.
     * gearSynergies: itemIds index dashboard.items; timeReductionPct vs global avg when combo present.
     * itemEffectiveness: per-item timeReductionVsGlobalPct (positive = shorter fights vs baseline).
     * status: used by GENERAL analytics only — not shown on BOSSES tab.
     */
    bosses: [
      {
        id: 1,
        name: 'Malenia',
        lifespan: '05:20',
        status: 'Defeated',
        globalLifespanSamples: ['05:08', '06:01'],
        gearSynergies: [
          { itemIds: [2, 1], timeReductionPct: 19 },
          { itemIds: [2, 3], timeReductionPct: 11 },
        ],
        itemEffectiveness: [
          { itemId: 2, timeReductionVsGlobalPct: 22 },
          { itemId: 1, timeReductionVsGlobalPct: 14 },
          { itemId: 3, timeReductionVsGlobalPct: -5 },
        ],
      },
      {
        id: 2,
        name: 'Radahn',
        lifespan: '02:15',
        status: 'Alive',
        globalLifespanSamples: ['00:45', '01:12'],
        gearSynergies: [
          { itemIds: [2, 1], timeReductionPct: 16 },
          { itemIds: [1, 3], timeReductionPct: 7 },
        ],
        itemEffectiveness: [
          { itemId: 2, timeReductionVsGlobalPct: 17 },
          { itemId: 1, timeReductionVsGlobalPct: 9 },
          { itemId: 3, timeReductionVsGlobalPct: 3 },
        ],
      },
    ],
    /**
     * Analyzed sessions. bossEncounters: bossId, lifespan, optional loadout (1–3 item ids from dashboard.items).
     * Used to derive most lethal equipment synergy (lowest mean lifespan per loadout signature).
     */
    runsHistory: [
      {
        id: 'RUN-001',
        date: '2024-05-18',
        duration: '00:28:00',
        bossEncounters: [
          { bossId: 1, lifespan: '04:48', loadout: [2, 1] },
          { bossId: 2, lifespan: '01:52', loadout: [2, 7] },
        ],
      },
      {
        id: 'RUN-002',
        date: '2024-05-19',
        duration: '00:45:30',
        bossEncounters: [{ bossId: 1, lifespan: '05:55', loadout: [3, 11] }],
      },
      {
        id: 'RUN-003',
        date: '2024-05-20',
        duration: '00:30:00',
        bossEncounters: [
          { bossId: 1, lifespan: '05:20', loadout: [2] },
          { bossId: 2, lifespan: '02:15', loadout: [11] },
        ],
      },
      {
        id: 'RUN-004',
        date: '2024-05-20',
        duration: '00:52:10',
        bossEncounters: [{ bossId: 2, lifespan: '02:40', loadout: [3, 8] }],
      },
      {
        id: 'RUN-005',
        date: '2024-05-21',
        duration: '01:10:00',
        bossEncounters: [
          { bossId: 1, lifespan: '06:02', loadout: [8, 6] },
          { bossId: 2, lifespan: '01:58', loadout: [2, 1] },
        ],
      },
      {
        id: 'RUN-006',
        date: '2024-05-22',
        duration: '00:38:45',
        bossEncounters: [
          { bossId: 1, lifespan: '04:31', loadout: [2, 1] },
          { bossId: 2, lifespan: '02:22', loadout: [9, 4] },
        ],
      },
    ],
  },
};

export function cloneInitialData() {
  return structuredClone(initialData);
}
