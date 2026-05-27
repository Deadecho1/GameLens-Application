import { useMemo } from 'react';
import {
  formatCountDelta,
  formatDurationSecondsDelta,
  formatPercentPointDelta,
} from '../../utils/analyticsDelta';

/**
 * Version B split-view delta vs Version A baseline.
 * @param {'count'|'duration'|'percent'} kind
 * @param {number|null|undefined} baseline — omit or null to hide
 * @param {number|null|undefined} current
 */
export default function DeltaIndicator({ kind, baseline, current }) {
  const formatted = useMemo(() => {
    if (baseline === undefined || baseline === null) return null;
    if (current === undefined || current === null) return null;

    switch (kind) {
      case 'count':
        return formatCountDelta(baseline, current);
      case 'duration':
        return formatDurationSecondsDelta(baseline, current);
      case 'percent':
        return formatPercentPointDelta(baseline, current);
      default:
        return null;
    }
  }, [kind, baseline, current]);

  if (!formatted) return null;

  const colorClass =
    formatted.direction === 'up'
      ? 'text-emerald-400'
      : formatted.direction === 'down'
        ? 'text-rose-400'
        : 'text-slate-300';

  return (
    <p
      className={`font-data mt-1.5 text-sm font-medium tabular-nums tracking-tight ${colorClass}`}
      aria-label={`Change vs version A: ${formatted.text}`}
    >
      {formatted.text}
    </p>
  );
}
