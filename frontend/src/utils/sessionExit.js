/**
 * Normalizes app state when leaving the main session (logout or guest exit → Welcome).
 */
export function mergeExitSessionState(prev, ipcState = null) {
  const base = ipcState && typeof ipcState === 'object' ? ipcState : prev;
  const procStatus = base.processing?.status;
  let nextStatus = procStatus;
  if (procStatus === 'completed' || procStatus === 'running') {
    nextStatus = 'idle';
  }

  return {
    ...base,
    auth: {
      loggedIn: false,
      email: null,
      userId: null,
      syncStatus: 'idle',
      syncMessage: '',
    },
    ui: {
      ...(base.ui ?? {}),
      postProcessingReviewOpen: false,
      completionCelebrationActive: false,
      addGameModalOpen: false,
      addVersionModalOpen: false,
      changePicker: null,
    },
    processing: {
      ...(base.processing ?? {}),
      status: nextStatus ?? 'idle',
      pendingRun: null,
    },
  };
}

export const EXIT_UI_PATCH = {
  postProcessingReviewOpen: false,
  completionCelebrationActive: false,
  addGameModalOpen: false,
  addVersionModalOpen: false,
  changePicker: null,
};
