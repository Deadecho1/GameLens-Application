import { durationToSeconds } from './duration';

function finite(n) {
  const v = Number(n);
  return Number.isFinite(v) ? v : null;
}

/** Format duration delta as mm:ss with padded minutes (e.g. 03:09). */
export function formatDeltaDurationLabel(totalSec) {
  const s = Math.max(0, Math.round(Math.abs(totalSec)));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

/**
 * @returns {{ direction: 'up'|'down'|'neutral', text: string } | null}
 */
export function formatCountDelta(baseline, current) {
  const b = finite(baseline);
  const c = finite(current);
  if (b === null || c === null) return null;

  const diff = c - b;
  if (diff === 0) return { direction: 'neutral', text: '—' };

  const pct =
    b !== 0 ? Math.round((diff / b) * 100) : c > 0 ? 100 : 0;
  const pctLabel = `${pct > 0 ? '+' : ''}${pct}%`;

  return {
    direction: diff > 0 ? 'up' : 'down',
    text: `${diff > 0 ? '↑' : '↓'} ${Math.abs(diff)} (${pctLabel})`,
  };
}

/**
 * @returns {{ direction: 'up'|'down'|'neutral', text: string } | null}
 */
export function formatDurationSecondsDelta(baselineSec, currentSec) {
  const b = finite(baselineSec);
  const c = finite(currentSec);
  if (b === null || c === null) return null;

  const diff = c - b;
  if (diff === 0) return { direction: 'neutral', text: '—' };

  const absLabel = formatDeltaDurationLabel(diff);
  const pctPart =
    b > 0
      ? ` (${diff > 0 ? '+' : ''}${Math.round((diff / b) * 100)}%)`
      : '';

  return {
    direction: diff > 0 ? 'up' : 'down',
    text: `${diff > 0 ? '↑' : '↓'} ${absLabel}${pctPart}`,
  };
}

/**
 * Percentage-point difference (e.g. 42% → 47% = ↑ 5%).
 * @returns {{ direction: 'up'|'down'|'neutral', text: string } | null}
 */
export function formatPercentPointDelta(baselinePct, currentPct) {
  const b = finite(baselinePct);
  const c = finite(currentPct);
  if (b === null || c === null) return null;

  const diff = Math.round(c - b);
  if (diff === 0) return { direction: 'neutral', text: '—' };

  return {
    direction: diff > 0 ? 'up' : 'down',
    text: `${diff > 0 ? '↑' : '↓'} ${Math.abs(diff)}%`,
  };
}

/** Parse display duration strings (HH:MM:SS / MM:SS) for delta math. */
export function formatDurationStringDelta(baselineLabel, currentLabel) {
  return formatDurationSecondsDelta(
    durationToSeconds(baselineLabel),
    durationToSeconds(currentLabel),
  );
}
