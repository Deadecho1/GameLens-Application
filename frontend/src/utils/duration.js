/** Parse "HH:MM:SS" or "MM:SS" to total seconds. */
export function durationToSeconds(hms) {
  if (!hms || typeof hms !== 'string') return 0;
  const parts = hms.split(':').map((p) => parseInt(p, 10));
  if (parts.some((n) => Number.isNaN(n))) return 0;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parts[0] || 0;
}

/** Seconds to minutes (decimal) for charts. */
export function secondsToMinutes(sec) {
  return Math.round((sec / 60) * 100) / 100;
}

/** Format seconds as H:MM:SS or M:SS for tooltips. */
export function formatSecondsAsHMS(totalSec) {
  const s = Math.max(0, Math.floor(totalSec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  return `${m}:${String(sec).padStart(2, '0')}`;
}
