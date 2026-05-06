import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Columns2, GitCompare } from 'lucide-react';
import GeneralMissionStats from '../analytics/GeneralMissionStats';
import BossesAnalytics from '../analytics/BossesAnalytics';
import ItemsPowerLab from '../analytics/ItemsPowerLab';
import { initialData } from '../../dataStore.js';
import { sliceAnalyticsDataByVersion } from '../../utils/analyticsVersionSlice';

const SUB_TABS = [
  { id: 'general', label: 'GENERAL' },
  { id: 'bosses', label: 'BOSSES' },
  { id: 'items', label: 'ITEMS' },
];

function renderActiveSubView(sub, data) {
  if (sub === 'general') return <GeneralMissionStats data={data} />;
  if (sub === 'bosses') return <BossesAnalytics data={data} />;
  if (sub === 'items') return <ItemsPowerLab data={data} />;
  return null;
}

function VersionSelect({ id, label, value, versions, onChange, disabled }) {
  return (
    <label className="flex min-w-0 flex-col gap-1 sm:flex-row sm:items-center sm:gap-2">
      <span className="shrink-0 font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-500">
        {label}
      </span>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="min-w-0 flex-1 rounded-lg border border-slate-700/90 bg-slate-900/90 px-3 py-2 font-data text-xs text-slate-200 shadow-inner shadow-black/20 outline-none ring-cyan-500/0 transition focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {versions.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </label>
  );
}

/**
 * ANALYTICS — secondary nav + sub-views. Writes ui.analyticsSubTab.
 * Optional split-view compares two game versions (filtered `runsHistory` only).
 */
export default function AnalyticsTab({ data, onPatch }) {
  const sub = data.ui.analyticsSubTab;

  const versions = useMemo(() => {
    const list = data?.setup?.versions;
    if (Array.isArray(list) && list.length > 0) return list;
    return initialData.setup.versions ?? [];
  }, [data?.setup?.versions]);

  const [versionA, setVersionA] = useState(() => versions[0] ?? '');
  const [versionB, setVersionB] = useState(() => versions[1] ?? versions[0] ?? '');
  const [splitView, setSplitView] = useState(false);

  useEffect(() => {
    const a = versions[0] ?? '';
    const b = versions[1] ?? a;
    setVersionA((prev) => (versions.includes(prev) ? prev : a));
    setVersionB((prev) => (versions.includes(prev) ? prev : b));
  }, [versions]);

  const dataA = useMemo(
    () => sliceAnalyticsDataByVersion(data, splitView ? versionA : null),
    [data, splitView, versionA],
  );

  const dataB = useMemo(
    () => sliceAnalyticsDataByVersion(data, splitView ? versionB : null),
    [data, splitView, versionB],
  );

  const leftScrollRef = useRef(null);
  const rightScrollRef = useRef(null);
  const syncLock = useRef(false);

  const handleLeftScroll = useCallback((e) => {
    const right = rightScrollRef.current;
    if (!right || syncLock.current) return;
    syncLock.current = true;
    right.scrollTop = e.target.scrollTop;
    requestAnimationFrame(() => {
      syncLock.current = false;
    });
  }, []);

  const handleRightScroll = useCallback((e) => {
    const left = leftScrollRef.current;
    if (!left || syncLock.current) return;
    syncLock.current = true;
    left.scrollTop = e.target.scrollTop;
    requestAnimationFrame(() => {
      syncLock.current = false;
    });
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className="mx-auto max-w-[1800px] px-4 py-8 md:py-10"
    >
      <header className="mb-6">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Intelligence
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Analytics deck
        </h2>
      </header>

      <div
        className="sticky top-0 z-30 mb-6 border-b border-slate-800/80 bg-slate-950/92 px-1 py-3 shadow-[0_8px_24px_-8px_rgba(0,0,0,0.45)] backdrop-blur-md md:px-2"
        aria-label="Version comparison"
      >
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-2 text-slate-400">
            <GitCompare className="h-4 w-4 shrink-0 text-cyan-500/80" aria-hidden />
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-slate-300">
              Version comparison
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-4">
            <VersionSelect
              id="analytics-compare-version-a"
              label="Compare version A"
              value={versionA}
              versions={versions}
              onChange={setVersionA}
              disabled={versions.length === 0}
            />
            <VersionSelect
              id="analytics-compare-version-b"
              label="Compare version B"
              value={versionB}
              versions={versions}
              onChange={setVersionB}
              disabled={versions.length === 0}
            />
            <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2 transition hover:border-slate-600">
              <Columns2 className="h-4 w-4 text-slate-500" aria-hidden />
              <span
                id="analytics-split-view-label"
                className="font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-400"
              >
                Enable split view
              </span>
              <button
                type="button"
                role="switch"
                id="analytics-split-view"
                aria-labelledby="analytics-split-view-label"
                aria-checked={splitView}
                onClick={() => setSplitView((s) => !s)}
                className={`relative h-6 w-11 shrink-0 rounded-full border transition ${
                  splitView
                    ? 'border-cyan-500/50 bg-cyan-500/20'
                    : 'border-slate-600 bg-slate-800/80'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-slate-200 shadow transition-transform ${
                    splitView ? 'left-5 translate-x-0.5' : 'left-0.5'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
        {versions.length === 0 ? (
          <p className="font-data mt-2 text-[11px] text-amber-200/80">
            No versions in setup — add versions under Mission setup to enable comparison.
          </p>
        ) : null}
      </div>

      <nav
        className="mb-8 flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-950/40 p-2 backdrop-blur-md"
        aria-label="Analytics sections"
      >
        {SUB_TABS.map((t) => {
          const active = sub === t.id;
          const itemsTab = t.id === 'items';
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={`relative flex-1 rounded-lg px-4 py-2.5 font-display text-[10px] font-bold tracking-[0.2em] transition sm:flex-none sm:min-w-[100px] ${
                active
                  ? itemsTab
                    ? 'bg-violet-500/15 text-violet-200 ring-1 ring-violet-500/40'
                    : 'bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/35'
                  : 'text-slate-500 hover:bg-slate-900/60 hover:text-slate-300'
              }`}
              onClick={() =>
                onPatch({
                  ui: { ...data.ui, analyticsSubTab: t.id },
                })
              }
            >
              {t.label}
            </button>
          );
        })}
      </nav>

      <motion.div
        key={splitView ? `${sub}-split` : sub}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22 }}
        className={splitView ? 'min-h-0' : ''}
      >
        {!splitView ? (
          renderActiveSubView(sub, data)
        ) : (
          <div className="relative flex min-h-[min(640px,calc(100vh-13rem))] flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0">
            <div
              ref={leftScrollRef}
              onScroll={handleLeftScroll}
              className="min-h-[min(480px,55vh)] min-w-0 flex-1 overflow-y-auto overflow-x-hidden overscroll-contain lg:min-h-[min(640px,calc(100vh-14rem))] lg:max-h-[calc(100vh-14rem)] lg:w-1/2 lg:border-r lg:border-slate-800/80 lg:pr-4"
            >
              <div className="mb-2 border-b border-slate-800/60 pb-2">
                <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-cyan-500/90">
                  {versionA || 'Version A'}
                </p>
              </div>
              <div className="min-w-0">{renderActiveSubView(sub, dataA)}</div>
            </div>

            <div className="relative flex shrink-0 items-center justify-center py-2 lg:hidden">
              <div className="h-px w-full bg-gradient-to-r from-transparent via-slate-600/80 to-transparent" />
              <span className="absolute left-1/2 inline-flex -translate-x-1/2 rounded-full border border-slate-600/90 bg-slate-900/95 px-2.5 py-1 font-display text-[9px] font-bold tracking-[0.25em] text-slate-300 shadow-lg shadow-black/40 ring-1 ring-slate-800/90">
                VS
              </span>
            </div>

            <div
              ref={rightScrollRef}
              onScroll={handleRightScroll}
              className="min-h-[min(480px,55vh)] min-w-0 flex-1 overflow-y-auto overflow-x-hidden overscroll-contain lg:min-h-[min(640px,calc(100vh-14rem))] lg:max-h-[calc(100vh-14rem)] lg:w-1/2 lg:pl-4"
            >
              <div className="mb-2 border-b border-slate-800/60 pb-2">
                <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-indigo-300/90">
                  {versionB || 'Version B'}
                </p>
              </div>
              <div className="min-w-0">{renderActiveSubView(sub, dataB)}</div>
            </div>

            <div
              className="pointer-events-none absolute inset-y-4 left-1/2 z-20 hidden w-0 -translate-x-1/2 lg:block"
              aria-hidden
            >
              <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-gradient-to-b from-transparent via-slate-600/85 to-transparent" />
              <div className="absolute left-1/2 top-[38%] -translate-x-1/2 -translate-y-1/2">
                <span className="inline-flex rounded-full border border-slate-600/90 bg-slate-900/95 px-2.5 py-1 font-display text-[9px] font-bold tracking-[0.25em] text-slate-300 shadow-lg shadow-black/40 ring-1 ring-slate-800/90">
                  VS
                </span>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
