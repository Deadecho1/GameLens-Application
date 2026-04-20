import React, { useState } from 'react';

function App() {
  const [games, setGames] = useState(["Elden Ring", "Hades"]);
  const [versions, setVersions] = useState(["v1.0.1", "v1.1.0"]);
  const [selectedGame, setSelectedGame] = useState(games[0]);
  const [selectedVersion, setSelectedVersion] = useState(versions[0]);

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Top Bar */}
      <header className="p-4 bg-slate-800 border-b border-slate-700 flex items-center justify-between shadow-2xl">
        <div className="flex items-center gap-6">
          <h1 className="text-2xl font-black text-blue-500 tracking-tighter">GAMELENS</h1>
          
          {/* Game Dropdown */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-bold text-slate-400 uppercase">Game</label>
            <select 
              className="bg-slate-700 border border-slate-600 rounded px-3 py-1.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              value={selectedGame}
              onChange={(e) => setSelectedGame(e.target.value)}
            >
              {games.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
            <button className="bg-slate-600 hover:bg-blue-600 p-1.5 rounded transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8 4a.5.5 0 0 1 .5.5v3h3a.5.5 0 0 1 0 1h-3v3a.5.5 0 0 1-1 0v-3h-3a.5.5 0 0 1 0-1h3v-3A.5.5 0 0 1 8 4z"/>
              </svg>
            </button>
          </div>

          {/* Version Dropdown */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-bold text-slate-400 uppercase">Version</label>
            <select 
              className="bg-slate-700 border border-slate-600 rounded px-3 py-1.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
            >
              {versions.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
            <button className="bg-slate-600 hover:bg-blue-600 p-1.5 rounded transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8 4a.5.5 0 0 1 .5.5v3h3a.5.5 0 0 1 0 1h-3v3a.5.5 0 0 1-1 0v-3h-3a.5.5 0 0 1 0-1h3v-3A.5.5 0 0 1 8 4z"/>
              </svg>
            </button>
          </div>
        </div>

        <button className="bg-blue-600 hover:bg-blue-500 px-8 py-2 rounded-full font-bold uppercase text-sm tracking-widest transition-all shadow-[0_0_15px_rgba(37,99,235,0.4)]">
          Process Clip
        </button>
      </header>

      {/* Placeholder for Content */}
      <main className="p-12">
        <div className="max-w-4xl mx-auto border-2 border-dashed border-slate-800 rounded-3xl h-96 flex items-center justify-center text-slate-600">
          <p className="text-lg">Select game and version to begin analysis</p>
        </div>
      </main>
    </div>
  );
}

export default App;