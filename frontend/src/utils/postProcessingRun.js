/**
 * Pending run extraction and library sync after post-processing review.
 */

function todayIsoDate() {
  return new Date().toISOString().slice(0, 10);
}

function nextRunId(existingRuns) {
  const maxNum = (existingRuns ?? []).reduce((m, r) => {
    const match = /^RUN-(\d+)$/i.exec(String(r?.id ?? ''));
    return match ? Math.max(m, parseInt(match[1], 10)) : m;
  }, 0);
  return `RUN-${String(maxNum + 1).padStart(3, '0')}`;
}

function videoStem(fileName) {
  const base = String(fileName ?? 'session').split(/[/\\]/).pop() ?? 'session';
  return base.replace(/\.[^.]+$/, '') || 'session';
}

function synthesizePendingRun(data) {
  const files = data?.processing?.videoFiles ?? [];
  const lastFile = files.length ? files[files.length - 1] : null;
  const baseline = data?.dashboard?.runsHistory ?? [];
  return {
    id: nextRunId(baseline),
    date: todayIsoDate(),
    duration: '00:00:00',
    outcome: 'completed',
    bossEncounters: [],
    runName: videoStem(lastFile),
    _synthetic: true,
  };
}

/**
 * Prefer a run that appeared after the pre-completion snapshot; else newest; else synthesize.
 */
export function extractPendingRunFromState(data, baselineRuns = []) {
  const runs = Array.isArray(data?.dashboard?.runsHistory)
    ? data.dashboard.runsHistory
    : [];
  const baselineIds = new Set((baselineRuns ?? []).map((r) => String(r?.id)));

  const newcomers = runs.filter((r) => r?.id != null && !baselineIds.has(String(r.id)));
  if (newcomers.length > 0) {
    return newcomers[0];
  }

  if (runs.length > 0) {
    return runs[0];
  }

  return synthesizePendingRun(data);
}

export function confirmPendingRunToLibrary(data, pendingRun, baselineRuns = []) {
  if (!pendingRun) {
    return {
      ...data,
      ui: { ...data.ui, postProcessingReviewOpen: false },
      processing: {
        ...data.processing,
        status: 'idle',
        pendingRun: null,
      },
    };
  }

  const existing = Array.isArray(data?.dashboard?.runsHistory)
    ? data.dashboard.runsHistory
    : [];
  const hasId = existing.some((r) => String(r?.id) === String(pendingRun.id));
  const runsHistory = hasId ? existing : [...existing, pendingRun];

  return {
    ...data,
    ui: { ...data.ui, postProcessingReviewOpen: false },
    processing: {
      ...data.processing,
      status: 'idle',
      pendingRun: null,
    },
    dashboard: {
      ...data.dashboard,
      runsHistory,
    },
  };
}

export function discardPendingRun(data, pendingRun, baselineRuns = []) {
  const existing = Array.isArray(data?.dashboard?.runsHistory)
    ? data.dashboard.runsHistory
    : [];
  const baselineIds = new Set((baselineRuns ?? []).map((r) => String(r?.id)));
  const pendingId = pendingRun?.id != null ? String(pendingRun.id) : null;

  const runsHistory = existing.filter((r) => {
    if (pendingId && String(r?.id) === pendingId) return false;
    return true;
  });

  const restored =
    runsHistory.length > 0 || baselineRuns.length === 0
      ? runsHistory
      : [...baselineRuns];

  return {
    ...data,
    ui: { ...data.ui, postProcessingReviewOpen: false },
    processing: {
      ...data.processing,
      status: 'idle',
      pendingRun: null,
    },
    dashboard: {
      ...data.dashboard,
      runsHistory: restored,
    },
  };
}
