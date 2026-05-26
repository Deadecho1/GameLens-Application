import { Timer } from 'lucide-react';
import ItemsSubTabToggle from './items/ItemsSubTabToggle';
import ItemsBuildPanel from './items/ItemsBuildPanel';
import ItemsInformationPanel from './items/ItemsInformationPanel';

/**
 * ITEMS analytics — BUILD (survival simulator) + INFORMATION (master-detail intel).
 * Sub-tab state is controlled by AnalyticsTab so split view stays in sync.
 */
export default function ItemsPowerLab({
  data,
  compact = false,
  compareBaseline = null,
  itemsSubTab,
  onItemsSubTabChange,
}) {
  const catalog = data?.dashboard?.items ?? [];
  const runsHistory = data?.dashboard?.runsHistory ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <header className="min-w-0">
          <div className="flex items-center gap-2">
            <Timer className="h-5 w-5 text-violet-400" strokeWidth={1.25} aria-hidden />
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-violet-400/80">
              {itemsSubTab === 'build' ? 'Survival simulator' : 'Item intelligence'}
            </p>
          </div>
          <h3
            className={`mt-2 font-display font-bold text-slate-100 ${
              compact ? 'text-lg' : 'text-xl md:text-2xl'
            }`}
          >
            {itemsSubTab === 'build' ? 'Run duration lab' : 'Master-detail catalog'}
          </h3>
          <p className="font-data mt-1.5 text-sm text-slate-500">
            {itemsSubTab === 'build'
              ? 'Build estimate from per-item run averages in history'
              : 'Metrics derived from runs history and loadouts in application state'}
          </p>
        </header>
        <ItemsSubTabToggle value={itemsSubTab} onChange={onItemsSubTabChange} />
      </div>

      {itemsSubTab === 'build' ? (
        <ItemsBuildPanel
          data={data}
          compact={compact}
          compareBaseline={compareBaseline}
        />
      ) : (
        <ItemsInformationPanel
          catalog={catalog}
          runsHistory={runsHistory}
          compact={compact}
          compareBaseline={compareBaseline}
        />
      )}
    </div>
  );
}
