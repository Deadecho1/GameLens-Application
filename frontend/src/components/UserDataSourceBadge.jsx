/**
 * Small pill indicating cloud vs local data source.
 */
export default function UserDataSourceBadge({ badge, className = '' }) {
  if (!badge?.label) return null;

  const isOnline = badge.variant === 'online';

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 font-data text-[9px] font-semibold uppercase tracking-[0.12em] ${
        isOnline
          ? 'border-emerald-500/35 bg-emerald-500/10 text-emerald-300/90'
          : 'border-slate-600 bg-slate-900/60 text-slate-400'
      } ${className}`}
    >
      {isOnline ? (
        <span
          className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]"
          aria-hidden
        />
      ) : null}
      {badge.label}
    </span>
  );
}
