import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
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
import { formatSecondsAsHMS } from '../../../utils/duration';

const DEFAULT_PLOT_HEIGHT = 400;

const chartDateFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
});

function formatChartDate(unixTime) {
  if (!Number.isFinite(unixTime)) return '';
  return chartDateFormatter.format(new Date(unixTime));
}

function runIdFromRow(row) {
  if (!row) return '…';
  return row.run_id ?? row.runId ?? row.id ?? '…';
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

  const headerRef = useRef(null);
  const dataRef = useRef(data);
  dataRef.current = data;

  /** Remounts chart+brush on reset only — never tracks brush position in React state. */
  const [brushResetKey, setBrushResetKey] = useState(0);

  const yDomain = useMemo(() => safeYDomain(yMin, yMax, yPad), [yMin, yMax, yPad]);

  useEffect(() => {
    if (!headerRef.current) return;
    headerRef.current.textContent = '';
    headerRef.current.classList.add('invisible');
  }, [chartKey]);

  const updateBrushHeader = useCallback((e) => {
    const el = headerRef.current;
    const rows = dataRef.current;
    if (!e || !el || !rows?.length) return;

    const last = rows.length - 1;
    if (e.startIndex === 0 && e.endIndex >= last) {
      el.textContent = '';
      el.classList.add('invisible');
      return;
    }

    const startRow = rows[e.startIndex];
    const endRow = rows[e.endIndex];
    const startDate = dateLabelFromRow(startRow);
    const endDate = dateLabelFromRow(endRow);
    const startRunId = runIdFromRow(startRow);
    const endRunId = runIdFromRow(endRow);
    el.textContent = `${startDate} – ${endDate} · Run ${startRunId}–${endRunId}`;
    el.classList.remove('invisible');
  }, []);

  const handleResetZoom = useCallback(() => {
    if (headerRef.current) {
      headerRef.current.textContent = '';
      headerRef.current.classList.add('invisible');
    }
    setBrushResetKey((k) => k + 1);
  }, []);

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

  const canRenderChart = dataLength > 0 && plotHeightPx > 0;

  return (
    <div className={`flex w-full min-w-0 flex-col ${className ?? ''}`}>
      {n > 1 ? (
        <div className="mb-2 flex shrink-0 flex-wrap items-center justify-end gap-2 px-1">
          <span
            ref={headerRef}
            className="invisible min-h-[1em] font-data text-[10px] tabular-nums text-cyan-500/80 empty:invisible"
            aria-live="polite"
          />
          <button
            type="button"
            onClick={handleResetZoom}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1.5 font-display text-[8px] font-bold uppercase tracking-[0.15em] text-slate-400 transition hover:border-cyan-500/45 hover:text-cyan-200"
            aria-label="Reset chart zoom to show all runs"
          >
            <RotateCcw className="h-3 w-3" strokeWidth={1.5} aria-hidden />
            Reset zoom
          </button>
        </div>
      ) : null}

      <div
        className="w-full shrink-0"
        style={{ width: '100%', height: plotHeightPx, minHeight: plotHeightPx }}
      >
        {canRenderChart ? (
          <ResponsiveContainer width="100%" height={plotHeightPx} minHeight={plotHeightPx}>
            <ComposedChart
              key={`${chartKey}-${brushResetKey}`}
              data={data}
              margin={{ top: 16, right: 20, bottom: n > 1 ? 52 : 24, left: 12 }}
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
                offset: n > 1 ? -36 : -6,
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
            {n > 1 && dataLength > 1 ? (
              <Brush
                dataKey="timestamp"
                height={28}
                travellerWidth={10}
                stroke="rgba(34, 211, 238, 0.65)"
                fill="rgba(15, 23, 42, 0.94)"
                onChange={updateBrushHeader}
                tickFormatter={(unixTime) =>
                  new Intl.DateTimeFormat('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  }).format(new Date(unixTime))
                }
                isAnimationActive={false}
                alwaysShowText={false}
              />
            ) : null}
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
