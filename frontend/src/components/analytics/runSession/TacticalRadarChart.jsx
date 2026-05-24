import { memo, useCallback, useEffect, useMemo, useState } from 'react';
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
} from 'recharts';
import { formatSecondsAsHMS } from '../../../utils/duration';

const DEFAULT_PLOT_HEIGHT = 400;

const ZOOM_OPTIONS = [
  { id: 'ALL', label: 'All Time' },
  { id: 'LAST_50', label: 'Last 50 Runs' },
  { id: 'LAST_20', label: 'Last 20 Runs' },
];

const chartDateFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
});

function formatChartDate(unixTime) {
  if (!Number.isFinite(unixTime)) return '';
  return chartDateFormatter.format(new Date(unixTime));
}

function dateLabelFromRow(row) {
  if (!row) return '—';
  if (Number.isFinite(row.timestamp) && row.timestamp > 0) {
    const label = formatChartDate(row.timestamp);
    return label || '—';
  }
  if (row.date) return String(row.date);
  return '—';
}

function safeYDomain(yMin, yMax, yPad) {
  const lo = yMin - yPad;
  const hi = yMax + yPad;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [0, 3600];
  return [lo, hi];
}

function hexPolygonPoints(cx, cy, r) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 3) * i - Math.PI / 6;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(' ');
}

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

function RadarScatterShape({
  cx,
  cy,
  payload,
  selectedRunId,
  hoveredRunId,
  onSelectRun,
  onHoverRun,
}) {
  if (cx == null || cy == null || !payload) return null;
  const active = payload.runId === selectedRunId;
  const hover = payload.runId === hoveredRunId;
  const hr = active ? 10 : hover ? 9 : 8;
  const pts = hexPolygonPoints(cx, cy, hr);

  return (
    <g
      className="cursor-pointer"
      onClick={(e) => {
        e.stopPropagation();
        if (payload.runId != null) onSelectRun(payload.runId);
      }}
      onMouseEnter={() => onHoverRun(payload.runId, true)}
      onMouseLeave={() => onHoverRun(payload.runId, false)}
    >
      <polygon
        points={pts}
        fill={shardFill(active, hover)}
        stroke={shardStroke(active, hover)}
        strokeWidth={active ? 2 : hover ? 1.5 : 1}
        style={{
          filter:
            active || hover ? 'drop-shadow(0 0 5px rgba(34,211,238,0.55))' : 'none',
          transition: 'filter 0.15s ease, stroke-width 0.15s ease',
        }}
      />
    </g>
  );
}

function TacticalRadarChart({
  data,
  chartKey,
  n,
  yMin,
  yMax,
  yPad,
  globalAverageDurationSeconds,
  avgLabel,
  selectedRunId,
  hoveredRunId,
  onSelectRun,
  onHoverRun,
  className,
  plotHeight = DEFAULT_PLOT_HEIGHT,
}) {
  const dataLength = data?.length ?? 0;
  const plotHeightPx = Math.max(280, Number(plotHeight) || DEFAULT_PLOT_HEIGHT);

  const [zoomView, setZoomView] = useState('ALL');

  useEffect(() => {
    setZoomView('ALL');
  }, [chartKey]);

  const displayData = useMemo(() => {
    if (!data?.length) return [];
    if (zoomView === 'LAST_50') return data.slice(-50);
    if (zoomView === 'LAST_20') return data.slice(-20);
    return data;
  }, [data, zoomView]);

  const displayLength = displayData.length;

  const zoomRangeLabel = useMemo(() => {
    if (!displayLength || zoomView === 'ALL') return null;
    const start = dateLabelFromRow(displayData[0]);
    const end = dateLabelFromRow(displayData[displayLength - 1]);
    return `${start} – ${end} · ${displayLength} run${displayLength === 1 ? '' : 's'}`;
  }, [displayData, displayLength, zoomView]);

  const yDomain = useMemo(() => safeYDomain(yMin, yMax, yPad), [yMin, yMax, yPad]);

  const renderScatterShape = useCallback(
    (props) => (
      <RadarScatterShape
        {...props}
        selectedRunId={selectedRunId}
        hoveredRunId={hoveredRunId}
        onSelectRun={onSelectRun}
        onHoverRun={onHoverRun}
      />
    ),
    [selectedRunId, hoveredRunId, onSelectRun, onHoverRun],
  );

  const canRenderChart = displayLength > 0 && plotHeightPx > 0;

  return (
    <div className={`flex w-full min-w-0 flex-col ${className ?? ''}`}>
      {dataLength > 0 ? (
        <div className="mb-3 shrink-0 rounded-lg border border-slate-800 bg-slate-950/90 px-2 py-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-display text-[8px] font-bold uppercase tracking-[0.2em] text-slate-500">
              Tactical zoom
            </span>
            {zoomRangeLabel ? (
              <span className="font-data text-[10px] tabular-nums text-cyan-500/80">{zoomRangeLabel}</span>
            ) : null}
          </div>
          <div
            className="mt-2 flex flex-wrap gap-1.5"
            role="group"
            aria-label="Chart zoom range"
          >
            {ZOOM_OPTIONS.map((opt) => {
              const active = zoomView === opt.id;
              const minRuns = opt.id === 'LAST_50' ? 50 : opt.id === 'LAST_20' ? 20 : 0;
              const disabled = minRuns > 0 && dataLength < minRuns;
              return (
                <button
                  key={opt.id}
                  type="button"
                  disabled={disabled}
                  onClick={() => setZoomView(opt.id)}
                  className={`rounded-md border px-3 py-1.5 font-display text-[8px] font-bold uppercase tracking-[0.12em] transition ${
                    active
                      ? 'border-cyan-400/70 bg-cyan-500/15 text-cyan-100 shadow-[0_0_12px_rgba(34,211,238,0.25)]'
                      : 'border-slate-700 bg-slate-900/80 text-slate-400 hover:border-cyan-500/35 hover:text-cyan-200'
                  } disabled:cursor-not-allowed disabled:opacity-40`}
                  aria-pressed={active}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}

      <div
        className="w-full shrink-0"
        style={{ width: '100%', height: plotHeightPx, minHeight: plotHeightPx }}
      >
        {canRenderChart ? (
          <ResponsiveContainer width="100%" height={plotHeightPx} minHeight={plotHeightPx}>
            <ComposedChart
              key={chartKey}
              data={displayData}
              margin={{ top: 16, right: 20, bottom: 28, left: 12 }}
              isAnimationActive={false}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.45} />
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                minTickGap={32}
                interval="preserveStartEnd"
                tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                axisLine={{ stroke: '#475569' }}
                tickLine={{ stroke: '#475569' }}
                tickFormatter={(unixTime) =>
                  new Intl.DateTimeFormat('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  }).format(new Date(unixTime))
                }
                label={{
                  value: 'Session date',
                  position: 'insideBottom',
                  offset: -6,
                  fill: '#64748b',
                  fontSize: 10,
                }}
              />
              <YAxis
                type="number"
                dataKey="durationSec"
                domain={yDomain}
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
                  isAnimationActive={false}
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
                  const p = payload[0]?.payload;
                  if (!p) return null;
                  const runId = p.run_id ?? p.runId;
                  if (runId == null) return null;
                  const when = dateLabelFromRow(p);
                  return (
                    <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 shadow-xl">
                      <p className="font-display text-xs font-bold uppercase tracking-wide text-cyan-200">
                        Run ID · {runId}
                      </p>
                      <p className="font-data mt-1 text-[10px] tabular-nums text-slate-500">
                        <span className="text-cyan-300/90">{when}</span>
                        <span className="text-slate-600">
                          {' '}
                          · DB index {p.run_index ?? '—'}
                        </span>
                      </p>
                      <p className="font-data mt-2 text-sm tabular-nums text-white">
                        {p.durationLabel ?? formatSecondsAsHMS(p.durationSec)}
                      </p>
                      <p className="font-data text-[10px] tabular-nums text-slate-500">
                        {formatSecondsAsHMS(p.durationSec)}
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
                shape={renderScatterShape}
                isAnimationActive={false}
                animationDuration={0}
              />
              <Line
                type="monotone"
                dataKey="movingAverage"
                name="10-run trend"
                stroke="#67e8f9"
                strokeWidth={3.5}
                strokeOpacity={1}
                dot={false}
                activeDot={{
                  r: 5,
                  fill: '#ecfeff',
                  stroke: '#22d3ee',
                  strokeWidth: 2,
                  isAnimationActive: false,
                }}
                style={{ filter: 'drop-shadow(0 0 12px rgba(34,211,238,0.85))' }}
                isAnimationActive={false}
                animationDuration={0}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div
            className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/40 font-data text-sm text-slate-500"
            style={{ minHeight: plotHeightPx }}
          >
            No data to plot.
          </div>
        )}
      </div>
    </div>
  );
}

function radarChartPropsAreEqual(prev, next) {
  if (prev.data !== next.data) return false;
  if (prev.chartKey !== next.chartKey) return false;
  if (prev.plotHeight !== next.plotHeight) return false;
  if (prev.selectedRunId !== next.selectedRunId) return false;
  if (prev.hoveredRunId !== next.hoveredRunId) return false;
  return true;
}

export const MemoizedRadarChart = memo(TacticalRadarChart, radarChartPropsAreEqual);
