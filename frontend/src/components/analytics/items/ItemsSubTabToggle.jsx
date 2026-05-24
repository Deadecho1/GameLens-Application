const ITEMS_SUB_TABS = [
  { id: 'build', label: 'Build' },
  { id: 'information', label: 'Information' },
];

export default function ItemsSubTabToggle({ value, onChange }) {
  return (
    <div
      className="inline-flex gap-1 rounded-lg border border-slate-800/80 bg-slate-950/55 p-1 backdrop-blur-sm"
      role="tablist"
      aria-label="Items view"
    >
      {ITEMS_SUB_TABS.map((tab) => {
        const active = value === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(tab.id)}
            className={`rounded-md px-3 py-1.5 font-display text-[9px] font-bold uppercase tracking-[0.16em] transition ${
              active
                ? tab.id === 'information'
                  ? 'bg-violet-500/15 text-violet-200 ring-1 ring-violet-500/35'
                  : 'bg-cyan-500/10 text-cyan-200 ring-1 ring-cyan-500/30'
                : 'text-slate-500 hover:bg-slate-900/70 hover:text-slate-300'
            }`}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
