import { formatSecondsAsHMS } from '../../../utils/duration';
import { getItemFirstAppearanceSeconds } from '../../../utils/itemAnalytics';

const NO_DESCRIPTION = 'No description available.';

function itemDescription(item) {
  for (const key of ['description', 'lore']) {
    const value = item?.[key];
    if (value != null && String(value).trim()) {
      return String(value).trim();
    }
  }
  return NO_DESCRIPTION;
}

function resolveAcquiredLabel(item, run, itemId) {
  for (const key of ['acquiredAt', 'acquired_at']) {
    const value = item?.[key];
    if (value != null && String(value).trim()) {
      return String(value).trim();
    }
  }
  const seconds = run ? getItemFirstAppearanceSeconds(run, itemId) : null;
  if (seconds == null) return null;
  return formatSecondsAsHMS(seconds);
}

/**
 * Tactical item chip with hover tooltip for Run Session analytics.
 */
export default function RunItemChip({
  item,
  itemId,
  run = null,
  synergyBonusSeconds = null,
}) {
  const name =
    item?.name ?? (itemId != null ? `Item ${itemId}` : 'Unknown Item');
  const description = itemDescription(item);
  const acquiredLabel = resolveAcquiredLabel(item, run, itemId);

  return (
    <div className="group relative">
      <div
        className="relative flex cursor-default items-center justify-center rounded-sm border border-cyan-500/30 bg-cyan-500/5 px-3 py-1 shadow-[0_0_10px_rgba(34,211,238,0.1)] backdrop-blur-md transition-all duration-200 hover:border-cyan-400 hover:bg-cyan-500/10 group-hover:scale-105 group-hover:shadow-[0_0_14px_rgba(34,211,238,0.35)]"
        aria-describedby={itemId != null ? `run-item-tip-${itemId}` : undefined}
      >
        <div
          className="pointer-events-none absolute -left-px -top-px h-1.5 w-1.5 border-l border-t border-cyan-400"
          aria-hidden
        />
        <span className="font-data text-[10px] font-bold uppercase tracking-widest tabular-nums text-cyan-200/90 transition-colors group-hover:text-cyan-100">
          {name}
        </span>
      </div>

      <div
        id={itemId != null ? `run-item-tip-${itemId}` : undefined}
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-[120] mb-2 w-56 -translate-x-1/2 opacity-0 transition-opacity duration-200 group-hover:opacity-100"
      >
        <div className="rounded-md border border-slate-700 bg-slate-800 p-3 shadow-xl shadow-black/40">
          <p className="font-display text-xs font-bold uppercase tracking-wide text-emerald-400">
            {name}
          </p>
         
          {acquiredLabel ? (
            <p className="mt-2 font-data text-[10px] leading-snug text-slate-500">
              Acquired: {acquiredLabel}
            </p>
          ) : null}
        </div>
        <div
          className="absolute left-1/2 top-full h-0 w-0 -translate-x-1/2 border-x-[6px] border-t-[6px] border-x-transparent border-t-slate-700"
          aria-hidden
        />
      </div>

      {synergyBonusSeconds != null ? (
        <span className="pointer-events-none absolute -right-1 -top-1 rounded border border-cyan-500/40 bg-slate-950 px-1 font-data text-[9px] font-bold text-cyan-300 opacity-0 shadow-sm transition-opacity duration-200 group-hover:opacity-100">
          +{synergyBonusSeconds}s
        </span>
      ) : null}
    </div>
  );
}
