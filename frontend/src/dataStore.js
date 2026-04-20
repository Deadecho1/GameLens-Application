
export const initialData = {
  setup: {
    games: ["Elden Ring", "Hades", "Cyberpunk 2077"],
    versions: ["v1.0.1-stable", "v1.1.0-alpha"],
    selectedGame: "Elden Ring",
    selectedVersion: "v1.1.0-alpha"
  },

  processing: {
    pipelinePath: "", 
    videoFiles: [],  
    selectedOption: "only event", 
    status: "idle",  
    logs: []       
  },

  dashboard: {
    stats: {
      totalRuns: 0,
      averageRunTime: "00:00:00",
      longestRun: "00:00:00",
      totalItemsFound: 0
    },
    items: [
    ],
    bosses: [
    ],
    runsHistory: []
  }
};