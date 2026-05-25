import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { motion } from 'framer-motion';
import { Clock, Link2, Search, TrendingUp } from 'lucide-react';
import { computeItemDetailAnalytics } from '../../../utils/itemAnalytics';
import DeltaIndicator from '../DeltaIndicator';
import { itemAccentDotStyle } from './itemUi';

const PHASE_COLORS = ['#22d3ee', '#8b5cf6', '#f59e0b'];

export default function ItemsInformationPanel({
  catalog = [],
  runsHistory = [],
  compact = false,
  compareBaseline = null,
}) {
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState(null);

  const sortedCatalog = useMemo(
    () => [...catalog].sort((a, b) => String(a.name).localeCompare(String(b.name))),
    [catalog],
  );

  const filteredCatalog = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sortedCatalog;
    return sortedCatalog.filter((item) => String(item.name).toLowerCase().includes(q));
  }, [sortedCatalog, search]);

  useEffect(() => {
    if (filteredCatalog.length === 0) {
      setSelectedId(null);
      return;
    }
    const stillVisible = filteredCatalog.some((i) => Number(i.id) === Number(selectedId));
    if (!stillVisible) setSelectedId(filteredCatalog[0].id);
  }, [filteredCatalog, selectedId]);

  const detail = useMemo(
    () =>
      selectedId != null
        ? computeItemDetailAnalytics(catalog, runsHistory, selectedId)
        : null,
    [catalog, runsHistory, selectedId],
  );

  const baselineDetail = useMemo(() => {
    if (!compareBaseline || selectedId == null) return null;
    const baseCatalog = compareBaseline.dashboard?.items ?? [];
    const baseRuns = compareBaseline.dashboard?.runsHistory ?? [];
    const inBaseline = baseCatalog.some(
      (i) => Number(i.id) === Number(selectedId),
    );
    if (!inBaseline) return null;
    return computeItemDetailAnalytics(baseCatalog, baseRuns, selectedId);
  }, [compareBaseline, selectedId]);

  const masterWidth = compact ? 'w-full lg:w-[min(100%,220px)]' : 'w-full lg:w-[min(100%,280px)]';

  return (
    <div
      className={`flex min-h-[min(520px,70vh)] flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-950/50 lg:flex-row lg:gap-0 ${
        compact ? '' : 'min-h-[min(640px,78vh)]'
      }`}
    >
      <aside
        className={`flex shrink-0 flex-col border-slate-800/80 lg:border-r ${masterWidth} lg:bg-slate-950/60`}
      >
        <div className="border-b border-slate-800/80 p-3">
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Item catalog
          </p>
          <label className="relative mt-2 block">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter items…"
              className="w-full rounded-lg border border-slate-700 bg-slate-950/85 py-2 pl-9 pr-3 font-data text-sm text-slate-100 placeholder:text-slate-600 outline-none focus:border-cyan-500/40 focus:ring-1 focus:ring-cyan-500/20"
            />
          </label>
        </div>
        <ul className="analytics-split-scroll min-h-0 flex-1 space-y-1 overflow-y-auto p-2">
          {filteredCatalog.length === 0 ? (
            <li className="font-data py-8 text-center text-sm text-slate-500">No items match.</li>
          ) : (
            filteredCatalog.map((item) => {
              const active = Number(item.id) === Number(selectedId);
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedId(item.id)}
                    className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition ${
                      active
                        ? 'border-cyan-500/45 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.08)]'
                        : 'border-slate-800 bg-slate-900/35 hover:border-slate-600 hover:bg-slate-900/60'
                    }`}
                  >
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={itemAccentDotStyle(item.id, active)}
                      aria-hidden
                    />
                    <span className="min-w-0 flex-1">
                      <span className="font-display block truncate text-[11px] font-bold uppercase tracking-wide text-slate-200">
                        {item.name}
                      </span>
                      {item.popularity != null ? (
                        <span className="font-data text-[10px] text-slate-500">
                          {Math.round(item.popularity)}% popularity
                        </span>
                      ) : null}
                    </span>
                  </button>
                </li>
              );
            })
          )}
        </ul>
      </aside>

      <section className="analytics-split-scroll min-h-0 min-w-0 flex-1 overflow-y-auto p-4 md:p-6">
        {!detail ? (
          <div className="flex h-full min-h-[280px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">
            <p className="font-data text-sm text-slate-500">Select an item to view intelligence.</p>
          </div>
        ) : (
          <motion.div
            key={detail.item.id}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.22 }}
            className="space-y-6"
          >
            <header className="border-b border-slate-800/80 pb-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <span
                    className="mt-1.5 h-3 w-3 shrink-0 rounded-full"
                    style={itemAccentDotStyle(detail.item.id, true)}
                    aria-hidden
                  />
                  <div>
                    <h4 className="font-display text-lg font-bold uppercase tracking-wide text-slate-100 md:text-xl">
                      {detail.item.name}
                    </h4>
                    <p className="font-data mt-0.5 text-xs text-slate-500">
                      Derived from {detail.runCount} run{detail.runCount === 1 ? '' : 's'} in history
                    </p>
                  </div>
                </div>
                {detail.popularity != null ? (
                  <div className="rounded-xl border border-violet-500/25 bg-violet-500/10 px-4 py-2 text-right">
                    <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-violet-300/80">
                      Popularity
                    </p>
                    <p className="font-data text-2xl font-bold tabular-nums text-violet-200">
                      {Math.round(detail.popularity)}%
                    </p>
                    {baselineDetail?.popularity != null ? (
                      <DeltaIndicator
                        kind="percent"
                        baseline={Math.round(Number(baselineDetail.popularity))}
                        current={Math.round(Number(detail.popularity))}
                      />
                    ) : null}
                  </div>
                ) : null}
              </div>
            </header>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <Clock className="h-4 w-4 text-cyan-400/80" />
                  <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                    Avg run duration when picked
                  </p>
                </div>
                <p className="font-data text-3xl font-bold tabular-nums text-cyan-200">
                  {detail.avgRunDurationLabel}
                </p>
                {baselineDetail?.avgRunDurationSec != null &&
                detail.avgRunDurationSec != null ? (
                  <DeltaIndicator
                    kind="duration"
                    baseline={baselineDetail.avgRunDurationSec}
                    current={detail.avgRunDurationSec}
                  />
                ) : null}
                <p className="font-data mt-1 text-[11px] text-slate-600">
                  Mean session length for runs containing this item
                </p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-emerald-400/80" />
                  <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                    Pick events
                  </p>
                </div>
                <p className="font-data text-3xl font-bold tabular-nums text-slate-200">
                  {detail.totalPicks}
                </p>
                {baselineDetail ? (
                  <DeltaIndicator
                    kind="count"
                    baseline={baselineDetail.totalPicks}
                    current={detail.totalPicks}
                  />
                ) : null}
                <p className="font-data mt-1 text-[11px] text-slate-600">
                  First appearance per run (loadout or pickup record)
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-4 md:p-5">
              <div className="mb-4 flex items-center gap-2">
                <Link2 className="h-4 w-4 text-violet-400/80" />
                <h5 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-300">
                  Top synergies · commonly picked with
                </h5>
              </div>
              {detail.topSynergies.length === 0 ? (
                <p className="font-data text-sm text-slate-500">
                  No co-occurring items in shared runs yet.
                </p>
              ) : (
                <ul className="space-y-2">
                  {detail.topSynergies.map((syn, idx) => (
                      <li
                        key={syn.id}
                        className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2.5"
                      >
                        <span className="font-display w-6 text-center text-[10px] font-bold text-slate-600">
                          #{idx + 1}
                        </span>
                        <span
                          className="h-2 w-2 shrink-0 rounded-full"
                          style={itemAccentDotStyle(syn.id)}
                          aria-hidden
                        />
                        <span className="font-display min-w-0 flex-1 truncate text-[11px] font-bold uppercase tracking-wide text-slate-200">
                          {syn.name}
                        </span>
                        <span className="font-data shrink-0 text-xs tabular-nums text-cyan-400/90">
                          {syn.count} run{syn.count === 1 ? '' : 's'} · {syn.pct}%
                        </span>
                      </li>
                    ))}
                </ul>
              )}
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-4 md:p-5">
              <h5 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-300">
                Pick priority over time
              </h5>
              <p className="font-data mt-1 text-[11px] text-slate-600">
                When this item was acquired (seconds into run, or inferred from encounter order)
              </p>
              <div className={`mt-4 w-full min-w-0 ${compact ? 'h-[200px]' : 'h-[240px]'}`}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={detail.pickPhaseChart}
                    margin={{ top: 8, right: 8, left: 0, bottom: 24 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.45} vertical={false} />
                    <XAxis
                      dataKey="label"
                      tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={false}
                      interval={0}
                      angle={compact ? -18 : 0}
                      textAnchor={compact ? 'end' : 'middle'}
                      height={compact ? 48 : 32}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={false}
                      tickLine={false}
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
                      formatter={(value, _name, item) => [
                        `${value} pick${value === 1 ? '' : 's'}`,
                        item?.payload?.sublabel ?? '',
                      ]}
                    />
                    <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={56}>
                      {detail.pickPhaseChart.map((entry, index) => (
                        <Cell key={entry.id} fill={PHASE_COLORS[index % PHASE_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-3 flex flex-wrap gap-3">
                {detail.pickPhaseChart.map((bucket, index) => (
                  <span
                    key={bucket.id}
                    className="font-data inline-flex items-center gap-1.5 text-[10px] text-slate-500"
                  >
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: PHASE_COLORS[index] }}
                    />
                    {bucket.sublabel}: {bucket.count}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </section>
    </div>
  );
}
