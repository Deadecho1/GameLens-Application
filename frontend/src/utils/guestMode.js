export function readGuestModeContinued() {
  return false;
}

export function persistGuestModeContinued() {
  /* Keep guest mode in-memory only for current session. */
}

export function clearGuestModeContinued() {
  /* No-op: nothing persisted across launches. */
}
