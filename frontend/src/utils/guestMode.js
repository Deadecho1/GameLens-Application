/** Persists "Continue as Guest" so Welcome is not shown on every launch. */
export const GUEST_MODE_STORAGE_KEY = 'gamelens_continue_as_guest';

export function readGuestModeContinued() {
  try {
    return localStorage.getItem(GUEST_MODE_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

export function persistGuestModeContinued() {
  try {
    localStorage.setItem(GUEST_MODE_STORAGE_KEY, '1');
  } catch {
    /* private mode / blocked storage */
  }
}

export function clearGuestModeContinued() {
  try {
    localStorage.removeItem(GUEST_MODE_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
