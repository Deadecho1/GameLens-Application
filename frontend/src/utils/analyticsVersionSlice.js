/**
 * Build a shallow-cloned `data` tree with `dashboard.runsHistory` filtered by game version.
 * Runs may use `gameVersion` or `version` (legacy).
 */
export function getRunGameVersion(run) {
  if (!run || typeof run !== 'object') return null;
  return run.gameVersion ?? run.version ?? null;
}

export function filterRunsHistoryByVersion(runsHistory, version) {
  const runs = Array.isArray(runsHistory) ? runsHistory : [];
  if (!version) return runs;
  const anyTagged = runs.some((r) => getRunGameVersion(r) != null);
  if (!anyTagged) return runs;
  return runs.filter((r) => getRunGameVersion(r) === version);
}

export function sliceAnalyticsDataByVersion(data, version) {
  if (!data || typeof data !== 'object') return data;
  const dashboard = data.dashboard ?? {};
  const filtered = filterRunsHistoryByVersion(dashboard.runsHistory, version);
  return {
    ...data,
    dashboard: {
      ...dashboard,
      runsHistory: filtered,
    },
  };
}
