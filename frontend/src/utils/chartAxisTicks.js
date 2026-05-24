/**
 * Sparse run-index labels for wide tactical radar charts (1000+ runs).
 * @param {number} totalRuns
 * @param {number} maxLabels — target tick count on axis
 */
export function sparseRunOrderTicks(totalRuns, maxLabels = 12) {
  const n = Math.max(0, Math.floor(totalRuns));
  if (n <= 0) return [];
  if (n <= maxLabels) {
    return Array.from({ length: n }, (_, i) => i + 1);
  }
  const step = Math.ceil((n - 1) / (maxLabels - 1));
  const ticks = [1];
  for (let v = 1 + step; v < n; v += step) {
    ticks.push(v);
  }
  if (ticks[ticks.length - 1] !== n) ticks.push(n);
  return ticks;
}

/** Tick step from chart width (px) and run count — at least ~48px between labels. */
export function runOrderTickStep(totalRuns, chartWidthPx = 720) {
  const n = Math.max(1, totalRuns);
  const maxLabels = Math.max(2, Math.floor(chartWidthPx / 48));
  if (n <= maxLabels) return 1;
  return Math.max(1, Math.ceil(n / maxLabels));
}

/**
 * Format X tick: show label only on step boundaries + first/last in view.
 */
export function formatSparseOrderTick(value, tickStep, minOrder, maxOrder) {
  const v = Math.round(Number(value));
  if (!Number.isFinite(v)) return '';
  if (v === minOrder || v === maxOrder) return String(v);
  if (tickStep <= 1) return String(v);
  if ((v - minOrder) % tickStep === 0) return String(v);
  return '';
}
