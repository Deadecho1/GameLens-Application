import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Calendar,
  Clock,
  Crosshair,
  Package,
  RotateCcw,
  Swords,
  Timer,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Scatter,
  Line,
  ReferenceLine,
  Brush,
} from 'recharts';
import { durationToSeconds, formatSecondsAsHMS } from '../../../utils/duration';
import { buildRunDurationTrendSeries } from '../../../utils/runMovingAverage';

/** Flat-top hexagon corners for SVG polygon */
function hexPolygonPoints(cx, cy, r) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 3) * i - Math.PI / 6;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(' ');
}

/** Synergy bonus label from catalog popularity (dataStore) — scaled to seconds proxy. */
function itemSynergySeconds(item) {
  if (!item || typeof item.popularity !== 'number') return 1;
  return Math.max(1, Math.round(item.popularity / 12));
}

/**
 * Converts run `duration` strings from dataStore (e.g. "00:28:00") to seconds for charts.
 */
export function runDurationToSeconds(hms) {
  return durationToSeconds(hms);
}

function itemById(catalog, id) {
  return catalog.find((i) => i.id === id) ?? null;
}

function itemNameById(catalog, id) {
  return itemById(catalog, id)?.name ?? null;
}

function bossNameById(catalog, id) {
  return catalog.find((b) => b.id === id)?.name ?? null;
}

/** Density cloud: uniform low-opacity fills so overlaps read brighter. */
const DENSITY_CLOUD_FILL = 'rgba(34, 211, 238, 0.18)';
const DENSITY_CLOUD_STROKE = 'rgba(34, 211, 238, 0.12)';

function shardFill(isSelected, isHover) {
  if (isSelected) return '#22d3ee';
  if (isHover) return 'rgba(34, 211, 238, 0.55)';
  return DENSITY_CLOUD_FILL;
}

function shardStroke(isSelected, isHover) {
  if (isSelected) return '#a5f3fc';
  if (isHover) return 'rgba(103, 232, 249, 0.8)';
  return DENSITY_CLOUD_STROKE;
}

const glitchInjectVariants = {
  initial: { opacity: 0.96, x: 0, skewX: 0 },
  animate: {
    opacity: [1, 0.94, 1, 1],
    x: [0, -3, 2, 0],
    skewX: ['0deg', '-0.6deg', '0.4deg', '0deg'],
    transition: { duration: 0.24, times: [0, 0.15, 0.35, 1], ease: 'easeOut' },
  },
};

/**
 * Run Session Analytics — data from `dataStore.js` (`initialData` + live `data`).
 */
export default function RunSessionAnalytics({ data }) {
  const runsHistory = useMemo(() => {
    const runs = data?.dashboard?.runsHistory;
    return Array.isArray(runs) ? runs : [];
  }, [data]);

  const itemsCatalog = data?.dashboard?.items ?? [];
  const bossesCatalog = data?.dashboard?.bosses ?? [];

  const globalAverageDurationSeconds = useMemo(() => {
    if (!runsHistory.length) return 0;
    const sum = runsHistory.reduce((acc, r) => acc + runDurationToSeconds(r.duration), 0);
    return sum / runsHistory.length;
  }, [runsHistory]);

  const scatterAxis = useMemo(() => {
    if (!runsHistory.length) {
      return { scatterData: [], n: 0, yMin: 0, yMax: 1, yPad: 0 };
    }
    const trendSeries = buildRunDurationTrendSeries(runsHistory);
    const secs = trendSeries.map((p) => p.durationSec);
    const avgSecs = trendSeries.map((p) => p.movingAverage);
    const yMin = Math.min(...secs, ...avgSecs);
    const yMax = Math.max(...secs, ...avgSecs);
    const spread = yMax - yMin || 1;
    const yPad = Math.max(30, spread * 0.06);
    const n = trendSeries.length;
    const scatterData = trendSeries.map((point) => ({
      ...point,
      minSec: yMin,
      maxSec: yMax,
    }));
    return {
      scatterData,
      n,
      yMin,
      yMax,
      yPad,
    };
  }, [runsHistory]);

  /** Full chart series — moving average computed once per runsHistory change. */
  const { scatterData: chartData, n, yMin, yMax, yPad } = scatterAxis;

  const [brushRange, setBrushRange] = useState({ startIndex: 0, endIndex: 0 });

  const chartSeriesKey = useMemo(() => {
    if (!chartData.length) return '0';
    const first = chartData[0].runId;
    const last = chartData[chartData.length - 1].runId;
    return `${chartData.length}:${first}:${last}`;
  }, [chartData]);

  useEffect(() => {
    const end = Math.max(0, chartData.length - 1);
    setBrushRange({ startIndex: 0, endIndex: end });
  }, [chartSeriesKey, chartData.length]);

  const brushEndIndex = Math.min(
    Math.max(brushRange.endIndex, brushRange.startIndex),
    Math.max(0, n - 1),
  );
  const brushStartIndex = Math.min(brushRange.startIndex, brushEndIndex);

  const brushViewStart = chartData[brushStartIndex]?.run_index ?? 1;
  const brushViewEnd = chartData[brushEndIndex]?.run_index ?? n;
  const isBrushZoomed = n > 1 && (brushStartIndex > 0 || brushEndIndex < n - 1);

  const handleBrushChange = useCallback((range) => {
    if (range?.startIndex == null || range?.endIndex == null) return;
    setBrushRange({
      startIndex: range.startIndex,
      endIndex: range.endIndex,
    });
  }, []);

  const handleResetZoom = useCallback(() => {
    setBrushRange({ startIndex: 0, endIndex: Math.max(0, n - 1) });
  }, [n]);

  const [selectedRunId, setSelectedRunId] = useState(null);
  const [hoveredRunId, setHoveredRunId] = useState(null);

  useEffect(() => {
    if (runsHistory.length === 0) setSelectedRunId(null);
  }, [runsHistory]);

  const selectedRun = useMemo(
    () => runsHistory.find((r) => r.id === selectedRunId) ?? null,
    [runsHistory, selectedRunId]
  );

  const maxBossLifespanSec = useMemo(() => {
    const enc = selectedRun?.bossEncounters ?? [];
    if (!enc.length) return 1;
    return Math.max(...enc.map((e) => durationToSeconds(e.lifespan)), 1);
  }, [selectedRun]);

  const handleListSelect = useCallback((runId) => {
    setSelectedRunId(runId);
  }, []);

  const CustomScatterShape = useCallback(
    (props) => {
      const { cx, cy, payload } = props;
      if (cx == null || cy == null || !payload) return null;
      const active = payload.runId === selectedRunId;
      const hover = payload.runId === hoveredRunId;
      const hr = active ? 10 : hover ? 9 : 8;
      const fill = shardFill(active, hover);
      const stroke = shardStroke(active, hover);
      const strokeW = active ? 2 : hover ? 1.5 : 1;
      const glow =
        active || hover
          ? 'drop-shadow(0 0 5px rgba(34,211,238,0.55))'
          : 'none';
      const pts = hexPolygonPoints(cx, cy, hr);
      const onActivate = (e) => {
        e.stopPropagation();
        if (payload.runId != null) setSelectedRunId(payload.runId);
      };
      return (
        <g
          className="cursor-pointer"
          onClick={onActivate}
          onMouseEnter={() => setHoveredRunId(payload.runId)}
          onMouseLeave={() => setHoveredRunId((h) => (h === payload.runId ? null : h))}
        >
          <polygon
            points={pts}
            fill={fill}
            stroke={stroke}
            strokeWidth={strokeW}
            style={{ filter: glow, transition: 'filter 0.15s ease, stroke-width 0.15s ease' }}
          />
        </g>
      );
    },
    [selectedRunId, hoveredRunId]
  );

  const avgLabel = formatSecondsAsHMS(Math.round(globalAverageDurationSeconds));

  const chartIsHero = !selectedRunId;

  const circuitBgStyle = {
    backgroundColor: '',
    backgroundImage: `

    `,
    backgroundSize: '32px 32px, 32px 32px, 8px 8px, 8px 8px',
    backgroundPosition: '0 0, 0 0, -1px -1px, -1px -1px',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className="relative mx-auto max-w-[1800px] px-4 py-8 md:py-10"
      style={circuitBgStyle}
    >
      <header className="relative z-1 mb-6">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-cyan-500/70">
          Session intel
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Run session analytics
        </h2>
      </header>

      <div className="relative z-1 flex min-h-[min(640px,72vh)] flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0 lg:rounded-2xl lg:border lg:border-slate-800 lg:bg-slate-950/45 lg:shadow-[inset_0_1px_0_rgba(148,163,184,0.05)]">
        <aside className="flex w-full flex-col border-slate-800/80 lg:w-[min(100%,280px)] lg:shrink-0 lg:border-r lg:border-slate-800/90 lg:bg-slate-950/55">
          <div className="border-b border-slate-800/80 px-4 py-3">
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
              Run selection
            </p>
            <p className="font-data mt-1 text-[10px] text-slate-600">
              {runsHistory.length} session{runsHistory.length === 1 ? '' : 's'} (dataStore)
            </p>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-2 [scrollbar-color:rgba(71,85,105,0.45)_transparent]">
            {runsHistory.length === 0 ? (
              <p className="font-data px-2 py-8 text-center text-sm text-slate-500">No runs in history.</p>
            ) : (
              <ul className="space-y-1">
                {runsHistory.map((run) => {
                  const active = run.id === selectedRunId;
                  return (
                    <li key={run.id}>
                      <button
                        type="button"
                        onClick={() => handleListSelect(run.id)}
                        className={`flex w-full flex-col gap-1 rounded-lg border px-3 py-2.5 text-left transition ${
                          active
                            ? 'border-cyan-500/50 bg-cyan-500/5'
                            : 'border-slate-800 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/65'
                        }`}
                      >
                        <span className="font-display text-[11px] font-bold uppercase tracking-wide text-slate-200">
                          {run.id}
                        </span>
                        <span className="font-data flex items-center gap-1.5 text-[10px] text-slate-500">
                          <Calendar className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
                          {run.date}
                        </span>
                        <span className="font-data flex items-center gap-1.5 text-[11px] tabular-nums text-cyan-300/80">
                          <Clock className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
                          {run.duration}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>

        <motion.div layout className="flex min-w-0 flex-1 flex-col bg-slate-950/30">
          <motion.section
            layout
            transition={{ type: 'spring', stiffness: 320, damping: 32 }}
            className={`border-b border-slate-800/80 p-4 backdrop-blur-sm md:p-6 ${
              chartIsHero
                ? 'min-h-[min(52vh,520px)] lg:min-h-[min(56vh,560px)]'
                : 'min-h-[200px] md:min-h-[240px]'
            }`}
          >
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <Activity className="h-4 w-4 text-cyan-400/80" aria-hidden />
                <div>
                  <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-300">
                    Tactical radar · density cloud + trend
                  </h3>
                  <p className="font-data mt-1 text-[10px] text-slate-600">
                    Drag the range slider below to zoom · {n} run{n === 1 ? '' : 's'} chronological
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {isBrushZoomed ? (
                  <span className="font-data text-[10px] tabular-nums text-cyan-500/70">
                    Runs {brushViewStart}–{brushViewEnd} of {n}
                  </span>
                ) : null}
                {n > 1 ? (
                  <button
                    type="button"
                    onClick={handleResetZoom}
                    disabled={!isBrushZoomed}
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1.5 font-display text-[8px] font-bold uppercase tracking-[0.15em] text-slate-400 transition enabled:hover:border-cyan-500/45 enabled:hover:text-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label="Reset chart zoom to show all runs"
                  >
                    <RotateCcw className="h-3 w-3" strokeWidth={1.5} aria-hidden />
                    Reset zoom
                  </button>
                ) : null}
              </div>
              {selectedRunId ? (
                <button
                  type="button"
                  onClick={() => setSelectedRunId(null)}
                  className="font-display text-[8px] font-bold uppercase tracking-[0.15em] text-cyan-500/80 underline-offset-2 hover:text-cyan-300 hover:underline"
                >
                  Overview
                </button>
              ) : null}
            </div>
            {chartData.length === 0 ? (
              <div className="flex h-[280px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40 font-data text-sm text-slate-500">
                No data to plot.
              </div>
            ) : (
              <div
                className={`w-full min-w-0 rounded-xl border border-slate-800 bg-slate-950/70 ${
                  chartIsHero ? 'h-[min(48vh,480px)] md:h-[min(50vh,520px)]' : 'h-[min(220px,32vh)] md:h-[260px]'
                }`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={chartData}
                    margin={{ top: 16, right: 20, bottom: n > 1 ? 52 : 24, left: 12 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.45} />
                    <XAxis
                      type="number"
                      dataKey="run_index"
                      domain={['dataMin', 'dataMax']}
                      scale="linear"
                      minTickGap={36}
                      tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      allowDecimals={false}
                      tickFormatter={(v) => (Number.isFinite(v) ? String(Math.round(v)) : '')}
                      label={{
                        value: 'Run index (chronological)',
                        position: 'insideBottom',
                        offset: n > 1 ? -36 : -6,
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                    />
                    <YAxis
                      type="number"
                      dataKey="durationSec"
                      domain={[yMin - yPad, yMax + yPad]}
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      axisLine={{ stroke: '#475569' }}
                      tickLine={{ stroke: '#475569' }}
                      tickFormatter={(v) => `${Math.round(v / 60)}m`}
                      allowDataOverflow={false}
                      label={{
                        value: 'Duration (seconds)',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                    />
                    {globalAverageDurationSeconds > 0 ? (
                      <ReferenceLine
                        y={globalAverageDurationSeconds}
                        stroke="rgb(34, 211, 238)"
                        strokeDasharray="5 4"
                        strokeOpacity={0.45}
                        label={{
                          value: `Global avg · ${avgLabel}`,
                          position: 'insideTopRight',
                          fill: 'rgb(34, 211, 238)',
                          fontSize: 10,
                          opacity: 0.85,
                          fontFamily: 'JetBrains Mono, monospace',
                        }}
                      />
                    ) : null}
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3', stroke: '#64748b' }}
                      contentStyle={{
                        background: 'rgba(15,23,42,0.96)',
                        border: '1px solid #334155',
                        borderRadius: 8,
                        fontSize: 12,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const scatterEntry = payload.find((e) => e?.payload?.runId != null);
                        const p = scatterEntry?.payload ?? payload[0]?.payload;
                        if (!p?.runId) return null;
                        return (
                          <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl">
                            <p className="font-display text-xs font-bold uppercase tracking-wide text-cyan-200">
                              {p.runId}
                            </p>
                            <p className="font-data mt-1 text-sm tabular-nums text-white">
                              {p.durationLabel ?? formatSecondsAsHMS(p.durationSec)}
                            </p>
                            <p className="font-data text-[10px] tabular-nums text-slate-500">
                              {formatSecondsAsHMS(p.durationSec)} · run #{p.run_index ?? p.order}
                            </p>
                            {typeof p.movingAverage === 'number' ? (
                              <p className="font-data mt-2 border-t border-slate-800 pt-2 text-[10px] text-slate-500">
                                10-run trend{' '}
                                <span className="tabular-nums text-cyan-400/90">
                                  {formatSecondsAsHMS(Math.round(p.movingAverage))}
                                </span>
                              </p>
                            ) : null}
                          </div>
                        );
                      }}
                    />
                    <Scatter
                      name="Density cloud"
                      dataKey="durationSec"
                      fill="#22d3ee"
                      fillOpacity={0.18}
                      shape={CustomScatterShape}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="movingAverage"
                      name="10-run trend"
                      stroke="#67e8f9"
                      strokeWidth={3.5}
                      strokeOpacity={1}
                      dot={false}
                      activeDot={{ r: 5, fill: '#ecfeff', stroke: '#22d3ee', strokeWidth: 2 }}
                      style={{ filter: 'drop-shadow(0 0 12px rgba(34,211,238,0.85))' }}
                      isAnimationActive={false}
                    />
                    {n > 1 ? (
                      <Brush
                        dataKey="run_index"
                        height={28}
                        travellerWidth={10}
                        stroke="rgba(34, 211, 238, 0.65)"
                        fill="rgba(15, 23, 42, 0.94)"
                        startIndex={brushStartIndex}
                        endIndex={brushEndIndex}
                        onChange={handleBrushChange}
                        tickFormatter={(v) => `#${Math.round(v)}`}
                        alwaysShowText={false}
                      />
                    ) : null}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </motion.section>

          <motion.section
            layout
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="relative flex-1 overflow-hidden p-4 md:p-6"
          >
            <AnimatePresence mode="wait">
              {!selectedRun ? (
                <motion.div
                  key="hint"
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 12 }}
                  transition={{ duration: 0.28 }}
                  className="flex min-h-[120px] items-center justify-center rounded-xl border border-dashed border-slate-700/80 bg-slate-950/40 px-4 py-8"
                >
                  <p className="font-data text-center text-sm text-slate-500">
                    Select a run or a shard on the radar to inject{' '}
                    <span className="text-cyan-500/80">telemetry</span>.
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key={selectedRun.id}
                  variants={glitchInjectVariants}
                  initial="initial"
                  animate="animate"
                  exit={{ opacity: 0 }}
                  className="relative rounded-2xl border border-slate-800 bg-slate-950/50 p-5"
                >
                  <div className="mb-6 border-b border-slate-800/90 pb-4">
                    <p className="font-display text-xs font-bold uppercase tracking-widest text-slate-300">
                      {selectedRun.id}
                    </p>
                    <p className="font-data mt-2 text-sm text-slate-300">
                      <span className="text-slate-500">Date:</span>{' '}
                      <span className="tabular-nums text-slate-200">{selectedRun.date}</span>
                    </p>
                    <p className="font-data mt-1 text-sm text-slate-300">
                      <span className="text-slate-500">Duration:</span>{' '}
                      <span className="tabular-nums text-cyan-300/90">{selectedRun.duration}</span>
                      <span className="ml-2 text-xs text-slate-600">
                        ({runDurationToSeconds(selectedRun.duration)}s)
                      </span>
                    </p>
                  </div>

                  <div className="flex items-center gap-2 pb-4">
                    <Timer className="h-4 w-4 text-slate-400" aria-hidden />
                    <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                      Tactical Run Trace
                    </h3>
                  </div>

                  {(selectedRun.bossEncounters ?? []).length === 0 ? (
                    <p className="font-data text-sm text-slate-600">No boss encounters for this run.</p>
                  ) : (
                    <div className="relative flex gap-4 md:gap-6">
                      <div className="relative w-8 shrink-0 md:w-10" aria-hidden>
                        <svg
                          className="absolute inset-0 h-full w-full text-cyan-500/35"
                          preserveAspectRatio="none"
                          viewBox="0 0 16 200"
                        >
                          <motion.path
                            d="M 8 0 L 8 28 L 3 34 L 13 40 L 8 52 L 8 76 L 2 82 L 14 88 L 8 100 L 8 124 L 4 130 L 12 136 L 8 148 L 8 172 L 5 178 L 11 184 L 8 200"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.2"
                            vectorEffect="non-scaling-stroke"
                            initial={{ pathLength: 1, opacity: 0.35 }}
                            animate={{ opacity: [0.32, 0.5, 0.38, 0.32] }}
                            transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
                          />
                        </svg>
                        <div className="absolute bottom-0 left-1/2 top-0 w-px -translate-x-1/2 bg-slate-700/80" />
                      </div>

                      <ul className="relative min-w-0 flex-1 space-y-10">
                        {(selectedRun.bossEncounters ?? []).map((enc, idx) => {
                          const loadoutIds = enc.loadout ?? [];
                          const bossLabel = bossNameById(bossesCatalog, enc.bossId);
                          const bossLifeSec = durationToSeconds(enc.lifespan);
                          const barPct = Math.min(100, Math.round((bossLifeSec / maxBossLifespanSec) * 100));

                          return (
                            <li key={`${selectedRun.id}-enc-${idx}`} className="relative">
                              <p className="font-display pb-3 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                                Encounter {idx + 1}
                              </p>

                              <div className="space-y-4">
                                <div>
                                  <div className="mb-2 flex items-center gap-2">
                                    <Package className="h-3.5 w-3.5 text-cyan-500/70" aria-hidden />
                                    <span className="font-display text-[9px] font-bold uppercase tracking-wider text-cyan-200/80">
                                      Stage 2 · Synergy nodes
                                    </span>
                                  </div>
                                  {loadoutIds.length === 0 ? (
                                    <p className="font-data text-xs text-slate-600">No loadout recorded.</p>
                                  ) : (
                                    <div className="flex flex-wrap gap-3">
                                      {loadoutIds.map((itemId) => {
                                        const itemName = itemsCatalog.find(i => i.id === itemId)?.name || 'Unknown Item';
                                        const row = itemById(itemsCatalog, itemId);
                                        const nm = row?.name ?? null;
                                        const bonus = itemSynergySeconds(row);
                                        return (
                                          <div key={`${selectedRun.id}-enc-${idx}-syn-${itemId}`} className="group relative">
                                            <div
                                              className="flex h-20 w-20 cursor-default items-center justify-center transition duration-200 group-hover:scale-110 group-hover:shadow-[0_0_14px_rgba(34,211,238,0.35)]"
                                              title={nm ?? `Item ${itemId}`}
                                            >
                                        
                                              <div className="relative flex items-center justify-center px-3 py-1 border border-cyan-500/30 bg-cyan-500/5 rounded-sm backdrop-blur-md shadow-[0_0_10px_rgba(34,211,238,0.1)] transition-all hover:border-cyan-400 hover:bg-cyan-500/10 group">
                                              {/* Tactical corner accent */}
                                              <div className="absolute -left-[1px] -top-[1px] h-1.5 w-1.5 border-l border-t border-cyan-400" />
                                              
                                              <span className="font-data text-[10px] font-bold uppercase tracking-widest tabular-nums text-cyan-200/90 group-hover:text-cyan-100">
                                                {itemName}
                                              </span>
                                            </div>
                                            </div>
                                            <span className="pointer-events-none absolute -right-1 -top-1 rounded border border-cyan-500/40 bg-slate-950 px-1 font-data text-[9px] font-bold text-cyan-300 opacity-0 shadow-sm transition group-hover:opacity-100">
                                              +{bonus}s
                                            </span>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  )}
                                </div>

                                <div
                                  className="relative overflow-hidden rounded-xl border bg-emerald-500/10 text-slate-200 p-4"
                                  style={{
                                    backgroundImage: `repeating-linear-gradient(
                                      -12deg,
                                      transparent,
                                      transparent 3px,
                                      rgba(239,68,68,0.04) 3px,
                                      rgba(239,68,68,0.04) 4px
                                    )`,
                                  }}
                                >
                                  <div className="relative z-1">
                                    <div className="mb-3 flex items-center gap-2">
                                      <Swords className="h-3.5 w-3.5 text-red-400/90" aria-hidden />
                                      <span className="font-display text-[9px] font-bold uppercase tracking-wider text-red-200/90">
                                        Stage 3 · High-intensity zone
                                      </span>
                                    </div>
                                    <p className="font-data text-sm text-slate-200">
                                      <span className="inline-flex items-center gap-1.5">
                                        <Crosshair className="h-3.5 w-3.5 text-slate-500" aria-hidden />
                                        {bossLabel ?? `Boss ${enc.bossId}`}
                                      </span>
                                      <span className="ml-2 font-mono text-xs text-slate-500">
                                        bossId {enc.bossId}
                                      </span>
                                    </p>
                                    <p className="font-data mt-2 text-[10px] uppercase tracking-wider text-slate-500">
                                      Lifespan (survived)
                                    </p>
                                    <div className="mt-2 h-2 overflow-hidden rounded border border-slate-700 bg-slate-900/80">
                                      <div
                                        className="h-full rounded border-cyan-500/40 transition-[width] duration-500"
                                        style={{ width: `${barPct}%` }}
                                      />
                                    </div>
                                    <p className="font-data mt-1.5 tabular-nums text-xs text-red-200/85">
                                      {enc.lifespan ?? '—'}
                                    </p>

                                    <div className="mt-4 border-t border-red-500/20 pt-4">
                                      <p className="font-display text-[9px] font-bold uppercase tracking-wider text-slate-400">
                                        Gear trace (this fight)
                                      </p>
                                      {loadoutIds.length === 0 ? (
                                        <p className="font-data mt-2 text-xs text-slate-500">No items tagged.</p>
                                      ) : (
                                        <ul className="mt-2 space-y-1.5">
                                          {loadoutIds.map((itemId) => (
                                            <li
                                              key={`${selectedRun.id}-enc-${idx}-gear-${itemId}`}
                                              className="font-data flex flex-wrap items-baseline gap-x-2 text-sm text-slate-300"
                                            >
                                              <span className="tabular-nums text-cyan-400/80">{itemId}</span>
                                              <span className="text-slate-500">
                                                {itemNameById(itemsCatalog, itemId) ?? '—'}
                                              </span>
                                            </li>
                                          ))}
                                        </ul>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.section>
        </motion.div>
      </div>
    </motion.div>
  );
}
