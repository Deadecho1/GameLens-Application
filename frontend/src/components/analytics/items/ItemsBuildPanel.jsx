import { useCallback, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import { Package, RotateCcw, Search } from 'lucide-react';
import { durationToSeconds } from '../../../utils/duration';
import { analyzeRunsForItems, formatMmSs } from '../../../utils/itemBuildAnalytics';
import DeltaIndicator from '../DeltaIndicator';
import { itemAccentDotStyle } from './itemUi';

const DRAG_MIME = 'application/gamelens-item-id';

function formatSignedDeltaMmSs(deltaSeconds) {
  const sign = deltaSeconds >= 0 ? '+' : '−';
  const abs = Math.abs(Math.round(deltaSeconds));
  const m = Math.floor(abs / 60);
  const sec = abs % 60;
  return `${sign}${m}:${String(sec).padStart(2, '0')}`;
}

function estimateBuildAvgRunSeconds(equippedIds, itemRunStats, globalAvgSeconds) {
  const ids = equippedIds.filter((id) => id != null);
  if (ids.length === 0) return globalAvgSeconds;
  let sum = 0;
  for (const id of ids) {
    const st = itemRunStats.get(id);
    sum += st?.runCount ? st.avgEquippedRunSeconds : globalAvgSeconds;
  }
  return sum / ids.length;
}

function placeItemInSlots(slots, slotIndex, itemId) {
  const next = [...slots];
  const dup = next.findIndex((id, i) => id === itemId && i !== slotIndex);
  if (dup >= 0) next[dup] = null;
  next[slotIndex] = itemId;
  return next;
}

/** Existing item combination simulator UI. */
export default function ItemsBuildPanel({ data, compact = false, compareBaseline = null }) {
  const catalog = data.dashboard.items ?? [];
  const runsHistory = data.dashboard.runsHistory ?? [];

  const [buildSlots, setBuildSlots] = useState([null, null, null, null, null]);
  const [activeSlot, setActiveSlot] = useState(0);
  const [search, setSearch] = useState('');

  const { globalAvgSeconds, itemRunStats } = useMemo(
    () => analyzeRunsForItems(runsHistory),
    [runsHistory],
  );

  const baselineGlobalAvgSeconds = useMemo(() => {
    if (!compareBaseline) return null;
    const baselineRuns = compareBaseline.dashboard?.runsHistory ?? [];
    return analyzeRunsForItems(baselineRuns).globalAvgSeconds;
  }, [compareBaseline]);

  const filteredItems = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return catalog;
    return catalog.filter((item) => String(item.name).toLowerCase().includes(q));
  }, [catalog, search]);

  const equippedIds = useMemo(() => buildSlots.filter((id) => id != null), [buildSlots]);

  const buildAvgSeconds = useMemo(
    () => estimateBuildAvgRunSeconds(buildSlots, itemRunStats, globalAvgSeconds),
    [buildSlots, itemRunStats, globalAvgSeconds],
  );

  const survivalDeltaSeconds = buildAvgSeconds - globalAvgSeconds;

  const barChartData = useMemo(
    () => [
      { key: 'baseline', label: 'Global average time', minutes: globalAvgSeconds / 60 },
      { key: 'build', label: 'Estimated time with these items', minutes: buildAvgSeconds / 60 },
    ],
    [globalAvgSeconds, buildAvgSeconds],
  );

  const assignItem = useCallback((itemId, slotIndex) => {
    setBuildSlots((s) => placeItemInSlots(s, slotIndex, itemId));
  }, []);

  const clearSlot = useCallback((slotIndex) => {
    setBuildSlots((s) => {
      const next = [...s];
      next[slotIndex] = null;
      return next;
    });
  }, []);

  const onBrowserItemActivate = (item) => {
    const empty = buildSlots.findIndex((id) => id == null);
    if (empty >= 0) {
      setBuildSlots((s) => placeItemInSlots(s, empty, item.id));
      setActiveSlot(empty);
    } else {
      setBuildSlots((s) => placeItemInSlots(s, activeSlot, item.id));
    }
  };

  const onDragStartItem = (e, itemId) => {
    e.dataTransfer.setData(DRAG_MIME, String(itemId));
    e.dataTransfer.effectAllowed = 'copy';
  };

  const onSlotDrop = (e, slotIndex) => {
    e.preventDefault();
    const raw = e.dataTransfer.getData(DRAG_MIME);
    const id = parseInt(raw, 10);
    if (!Number.isFinite(id) || !catalog.some((i) => i.id === id)) return;
    assignItem(id, slotIndex);
  };

  const clearBuild = () => {
    setBuildSlots([null, null, null, null, null]);
  };

  const barColors = ['#64748b', '#8b5cf6'];

  return (
    <div
      className={`flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-950/50 shadow-[inset_0_1px_0_rgba(148,163,184,0.06)] ${
        compact ? 'min-h-0 lg:flex-col' : 'min-h-[min(720px,78vh)] lg:flex-row lg:gap-0'
      }`}
    >
      <aside
        className={`flex w-full flex-col border-slate-800/80 bg-slate-950/60 ${
          compact
            ? 'border-b pb-4 lg:border-b lg:border-r-0'
            : 'lg:w-[min(100%,300px)] lg:shrink-0 lg:border-r'
        }`}
      >
        <div className="border-b border-slate-800/80 px-4 py-3">
          <p className="font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-300">
            Item list
          </p>
          <label className="relative mt-3 block">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-300" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search items…"
              className="w-full rounded-lg border border-slate-700 bg-slate-950/85 py-2.5 pl-10 pr-3 font-data text-sm text-slate-100 placeholder:text-slate-400 outline-none focus:border-slate-500 focus:ring-1 focus:ring-slate-500/40"
            />
          </label>
        </div>
        <div className="analytics-split-scroll min-h-0 max-h-[min(40vh,320px)] flex-1 overflow-y-auto p-2 lg:max-h-none">
          {filteredItems.length === 0 ? (
            <p className="font-data py-8 text-center text-base text-slate-300">No matches.</p>
          ) : (
            <ul className="space-y-1">
              {filteredItems.map((item) => {
                const inBuild = buildSlots.includes(item.id);
                const st = itemRunStats.get(item.id);
                const runCount = st?.runCount ?? 0;
                const avgLine =
                  runCount > 0
                    ? `Average time when equipped: ${formatMmSs(st.avgEquippedRunSeconds)}`
                    : 'Average time when equipped: —';
                const tooltipLines = [
                  `Found in ${runCount} session${runCount === 1 ? '' : 's'}`,
                  avgLine,
                ].join('\n');

                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      draggable
                      title={tooltipLines}
                      onDragStart={(e) => onDragStartItem(e, item.id)}
                      onClick={() => onBrowserItemActivate(item)}
                      className={`flex w-full items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition ${
                        inBuild
                          ? 'border-violet-500/35 bg-violet-500/5'
                          : 'border-slate-800 bg-slate-900/35 hover:border-slate-600 hover:bg-slate-900/60'
                      }`}
                    >
                      <span
                        className="h-2 w-2 shrink-0 rounded-full"
                        style={itemAccentDotStyle(item.id, inBuild)}
                        aria-hidden
                      />
                      <span className="font-display min-w-0 flex-1 truncate text-sm font-bold uppercase tracking-wide text-slate-200">
                        {item.name}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>

      <main className="relative min-w-0 flex-1 p-4 backdrop-blur-sm md:p-6 lg:p-8">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-10%,rgba(109,40,217,0.08),transparent)]" />

        <div className="relative space-y-8">
          <section>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Package className="h-4 w-4 text-slate-300" />
                <h4 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-300">
                  Selected items (max 5)
                </h4>
              </div>
              <button
                type="button"
                onClick={clearBuild}
                className="flex items-center gap-1.5 rounded-lg border border-slate-600 px-2.5 py-1.5 font-display text-sm font-bold uppercase tracking-[0.15em] text-slate-300 hover:border-slate-500 hover:text-slate-200"
              >
                <RotateCcw className="h-3 w-3" />
                Clear all items
              </button>
            </div>
            <p className="font-data mb-4 text-base text-slate-300">
              Drag or click an item to add it to a slot.
            </p>
            <div
              className={`grid gap-3 ${
                compact ? 'grid-cols-2 sm:grid-cols-3' : 'grid-cols-2 sm:grid-cols-3 md:grid-cols-5'
              }`}
            >
              {buildSlots.map((slotId, i) => {
                const item = slotId != null ? catalog.find((x) => x.id === slotId) : null;
                const isActive = activeSlot === i;
                return (
                  <div
                    key={i}
                    role="button"
                    tabIndex={0}
                    onClick={() => {
                      if (item) {
                        clearSlot(i);
                        return;
                      }
                      setActiveSlot(i);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        if (item) {
                          clearSlot(i);
                          return;
                        }
                        setActiveSlot(i);
                      }
                    }}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => onSlotDrop(e, i)}
                    className={`relative flex min-h-[120px] flex-col items-center justify-center rounded-xl border-2 border-dashed p-3 transition ${
                      isActive
                        ? 'border-violet-500/45 bg-violet-500/5 shadow-[0_0_16px_rgba(109,40,217,0.12)]'
                        : 'border-slate-700 bg-slate-950/50 hover:border-slate-600'
                    }`}
                  >
                    {item ? (
                      <>
                        <span
                          className="h-3 w-3 rounded-full"
                          style={itemAccentDotStyle(item.id, true)}
                          aria-hidden
                        />
                        <p className="font-display mt-3 max-w-full truncate px-1 text-center text-sm font-bold uppercase tracking-wide text-slate-200">
                          {item.name}
                        </p>
                        <button
                          type="button"
                          className="font-data absolute right-1 top-1 rounded px-1.5 text-sm font-medium text-slate-300 hover:text-slate-100"
                          onClick={(e) => {
                            e.stopPropagation();
                            clearSlot(i);
                          }}
                        >
                          ×
                        </button>
                      </>
                    ) : (
                      <span className="font-data text-sm font-semibold uppercase tracking-wider text-slate-300">
                        Slot {i + 1}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-950/40 p-5 backdrop-blur-md">
            <h4 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-300">
              Average session time
            </h4>
            <p className="font-data mt-1 text-base text-slate-300">
              Comparison between the global average time and the estimated time for this item combination.
            </p>
            <div className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
              <p className="font-data text-base text-slate-300">
                Global average time{' '}
                <span className="font-semibold tabular-nums text-slate-200">
                  {formatMmSs(globalAvgSeconds)}
                </span>
              </p>
              {baselineGlobalAvgSeconds != null ? (
                <DeltaIndicator
                  kind="duration"
                  baseline={baselineGlobalAvgSeconds}
                  current={globalAvgSeconds}
                />
              ) : null}
            </div>
            <div className="mt-4 h-[220px] w-full min-w-0 md:h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barChartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }} barGap={12}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} vertical={false} />
                  <XAxis
                    dataKey="label"
                    tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                    axisLine={{ stroke: '#475569' }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(v) => `${v}m`}
                    label={{
                      value: 'Minutes',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#64748b',
                      fontSize: 10,
                    }}
                  />
                  <RechartsTooltip
                    cursor={{ fill: 'rgba(148,163,184,0.06)' }}
                    contentStyle={{
                      background: 'rgba(15,23,42,0.95)',
                      border: '1px solid #334155',
                      borderRadius: 8,
                      fontSize: 12,
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                    labelStyle={{ color: '#e2e8f0' }}
                    formatter={(value) => [`${Number(value).toFixed(1)} min`, 'Average time']}
                  />
                  <Bar dataKey="minutes" radius={[6, 6, 0, 0]} maxBarSize={72}>
                    {barChartData.map((entry, index) => (
                      <Cell key={entry.key} fill={barColors[index]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <motion.div
              key={`${equippedIds.join('-')}-${globalAvgSeconds}`}
              initial={{ opacity: 0.92 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2 }}
              className="mt-6 border-t border-slate-800 pt-6"
            >
              <p className="font-display text-sm font-bold uppercase tracking-[0.25em] text-slate-300">
                Expected result
              </p>
              {!equippedIds.length ? (
                <p className="font-data mt-3 text-base text-slate-300">
                  Add at least one item to compare estimated time to the global average (
                  {formatMmSs(globalAvgSeconds)}).
                </p>
              ) : (
                <p
                  className={`font-data mt-3 text-2xl font-bold tabular-nums md:text-3xl ${
                    survivalDeltaSeconds > 0
                      ? 'text-emerald-400'
                      : survivalDeltaSeconds < 0
                        ? 'text-red-400'
                        : 'text-slate-200'
                  }`}
                >
                  {survivalDeltaSeconds > 0 && (
                    <>{formatSignedDeltaMmSs(survivalDeltaSeconds)} minutes added to session time</>
                  )}
                  {survivalDeltaSeconds < 0 && (
                    <>{formatSignedDeltaMmSs(survivalDeltaSeconds)} minutes below global average</>
                  )}
                  {survivalDeltaSeconds === 0 && (
                    <>Matches global average ({formatMmSs(globalAvgSeconds)})</>
                  )}
                </p>
              )}
              {equippedIds.length > 0 && (
                <p className="font-data mt-2 text-base text-slate-300">
                  Global average: {formatMmSs(globalAvgSeconds)} · Estimated time: {formatMmSs(buildAvgSeconds)}
                </p>
              )}
            </motion.div>
          </section>
        </div>
      </main>
    </div>
  );
}
