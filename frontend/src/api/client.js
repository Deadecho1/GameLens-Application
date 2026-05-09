/**
 * GameLens REST API client.
 *
 * Single swap point for mock vs real backend:
 *   - VITE_GAMELENS_MOCK=true  → returns data from mockData.json (no network calls)
 *   - VITE_COLLECTOR_URL       → base URL for the real Collector service (default: http://localhost:8000)
 *   - VITE_GAMELENS_USER_ID    → user identifier sent in X-User-ID header (default: dev-user)
 *
 * To point at a remote server, set VITE_COLLECTOR_URL before running Vite.
 */

import mockData from './mockData.json';

const IS_MOCK = import.meta.env.VITE_GAMELENS_MOCK === 'true';
const BASE_URL = (import.meta.env.VITE_COLLECTOR_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const DEFAULT_USER_ID = import.meta.env.VITE_GAMELENS_USER_ID ?? 'dev-user';

export function getUserId() {
  return DEFAULT_USER_ID;
}

async function apiFetch(path, params = {}, userId = DEFAULT_USER_ID) {
  const url = new URL(`${BASE_URL}${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v != null) url.searchParams.set(k, String(v));
  }
  const res = await fetch(url.toString(), {
    headers: { 'X-User-ID': String(userId) },
  });
  if (!res.ok) throw new Error(`Collector API ${res.status}: ${path}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Dashboard reads — called by analytics tabs
// ---------------------------------------------------------------------------

export async function getItems(userId, gameName, versionName) {
  if (IS_MOCK) return mockData.items;
  return apiFetch('/api/v1/dashboard/items', { game_name: gameName, version_name: versionName }, userId);
}

export async function getBosses(userId, gameName, versionName) {
  if (IS_MOCK) return mockData.bosses;
  return apiFetch('/api/v1/dashboard/bosses', { game_name: gameName, version_name: versionName }, userId);
}

export async function getRuns(userId, gameName, versionName) {
  if (IS_MOCK) return mockData.runsHistory;
  return apiFetch('/api/v1/dashboard/runs', { game_name: gameName, version_name: versionName }, userId);
}
