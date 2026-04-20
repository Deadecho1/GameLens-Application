
/**
 * GameLens Data Bridge
 * This file acts as the single source of truth between the Frontend and Backend.
 */

export const initialData = {
  // --- SETTINGS (Shared with Backend) ---
  setup: {
    games: ["Elden Ring", "Hades", "Cyberpunk 2077"],
    versions: ["v1.0.1-stable", "v1.1.0-alpha"],
    selectedGame: "Elden Ring",
    selectedVersion: "v1.1.0-alpha"
  },

  // --- PROCESSING STATE (Active during Run) ---
  processing: {
    pipelinePath: "C:/Games/Captures/EldenRing", // Backend reads this path
    videoFiles: ["boss_fight_01.mp4", "intro_cutscene.mp4"], // Backend populates this
    selectedOption: "only event", // "only event" | "only export" | "verbose"
    status: "idle",   // "idle" | "running" | "stopped" | "completed"
    logs: [
      "[INFO] System ready...",
      "[DEBUG] Waiting for user to click Run..."
    ] 
  },

  // --- DASHBOARD DATA (Populated after processing) ---
  dashboard: {
    stats: {
      totalRuns: 12,
      averageRunTime: "00:42:15",
      longestRun: "01:15:30",
      totalItemsFound: 154
    },
    items: [
      { id: 1, name: "Health Potion", popularity: 85, impact: "High" },
      { id: 2, name: "Iron Sword", popularity: 40, impact: "Medium" }
    ],
    bosses: [
      { id: 1, name: "Malenia", lifespan: "05:20", status: "Defeated" },
      { id: 2, name: "Radahn", lifespan: "02:15", status: "Alive" }
    ],
    runsHistory: [
      { id: "RUN-001", date: "2024-05-20", duration: "00:30:00" },
      { id: "RUN-002", date: "2024-05-21", duration: "01:10:00" }
    ]
  }
};