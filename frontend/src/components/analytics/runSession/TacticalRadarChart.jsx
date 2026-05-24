import { memo, useCallback } from 'react';
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
  onBrushUpdate,
  selectedRunId,
  hoveredRunId,
  onSelectRun,
  onHoverRun,
  className,
}) {
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

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          key={chartKey}
          data={data}
          margin={{ top: 16, right: 20, bottom: n > 1 ? 52 : 24, left: 12 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.45} />
          <XAxis
            dataKey="displayOrder"
            minTickGap={32}
            interval="preserveStartEnd"
            tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={{ stroke: '#475569' }}
            tickLine={{ stroke: '#475569' }}
            label={{
              value: 'Session order (1…n)',
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
                    Run ID · {p.runId}
                  </p>
                  <p className="font-data mt-1 text-[10px] tabular-nums text-slate-500">
                    DB run index ·{' '}
                    <span className="text-cyan-300/90">{p.run_index ?? '—'}</span>
                    <span className="text-slate-600">
                      {' '}
                      · slot {p.displayOrder} of {n}
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
              dataKey="displayOrder"
              height={28}
              travellerWidth={10}
              stroke="rgba(34, 211, 238, 0.65)"
              fill="rgba(15, 23, 42, 0.94)"
              onDragEnd={onBrushUpdate}
              tickFormatter={(v) => String(v)}
              alwaysShowText={false}
            />
          ) : null}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function radarChartPropsAreEqual(prev, next) {
  if (prev.data !== next.data) return false;
  if (prev.chartKey !== next.chartKey) return false;
  if (prev.selectedRunId !== next.selectedRunId) return false;
  if (prev.hoveredRunId !== next.hoveredRunId) return false;
  return true;
}

/** Isolated chart tree — parent zoom label updates must not re-render this SVG. */
export const MemoizedRadarChart = memo(TacticalRadarChart, radarChartPropsAreEqual);
