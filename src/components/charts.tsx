import React from "react";
import { fmt, pct } from "../lib/atlas";

export function Axis({
  x, y, w, h, xTicks, yTicks, xLabel, yLabel, xFormat, yFormat,
}: {
  x: (v: number) => number; y: (v: number) => number; w: number; h: number;
  xTicks: number[]; yTicks: number[]; xLabel?: string; yLabel?: string;
  xFormat?: (v: number) => string; yFormat?: (v: number) => string;
}) {
  const x0 = x(xTicks[0] ?? 0);
  return (
    <g className="axis">
      {yTicks.map((t, i) => (
        <g key={`y${i}`}>
          <line x1={x0} x2={w} y1={y(t)} y2={y(t)} className="gridline" />
          <line x1={x0 - 3} x2={x0} y1={y(t)} y2={y(t)} stroke="var(--rule-2)" />
          <text x={x0 - 6} y={y(t) + 3} textAnchor="end">{yFormat ? yFormat(t) : t}</text>
        </g>
      ))}
      <line x1={x0} x2={w} y1={h} y2={h} stroke="var(--rule-2)" />
      {xTicks.map((t, i) => (
        <g key={`x${i}`}>
          <line x1={x(t)} x2={x(t)} y1={h} y2={h + 3} stroke="var(--rule-2)" />
          <text x={x(t)} y={h + 12} textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}>
            {xFormat ? xFormat(t) : t}
          </text>
        </g>
      ))}
      {xLabel && <text x={w} y={h + 26} textAnchor="end" className="axislabel">{xLabel}</text>}
      {yLabel && (
        <text x={11} y={h / 2} transform={`rotate(-90 11 ${h / 2})`} textAnchor="middle" className="axislabel">
          {yLabel}
        </text>
      )}
    </g>
  );
}

export type Series = {
  values: number[]; color: string; label?: string; width?: number; dash?: string;
  opacity?: number; area?: boolean;
};

export function LineChart({
  width, height, series, yDomain, xLabel, yLabel, xTicks, yTicks,
  xFormat, yFormat, margin = { t: 12, r: 16, b: 28, l: 46 }, hoverIndex, onHover,
}: {
  width: number; height: number; series: Series[]; yDomain?: [number, number];
  xLabel?: string; yLabel?: string; xTicks?: number[]; yTicks?: number[];
  xFormat?: (v: number) => string; yFormat?: (v: number) => string;
  margin?: { t: number; r: number; b: number; l: number };
  hoverIndex?: number | null; onHover?: (i: number | null) => void;
}) {
  const n = Math.max(...series.map((s) => s.values.length), 2);
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const lo = yDomain?.[0] ?? Math.min(0, ...all);
  const hi = yDomain?.[1] ?? Math.max(...all, 0.001);
  const sx = (i: number) => margin.l + (i / (n - 1)) * (width - margin.l - margin.r);
  const sy = (v: number) => margin.t + (1 - (v - lo) / (hi - lo || 1)) * (height - margin.t - margin.b);
  const ticks = xTicks ?? Array.from({ length: 5 }, (_, i) => Math.round((i / 4) * (n - 1)));
  const yt = yTicks ?? Array.from({ length: 5 }, (_, i) => lo + (i / 4) * (hi - lo));
  const path = (vals: number[]) =>
    vals.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");

  return (
    <svg width={width} height={height} className="chart" onMouseLeave={() => onHover?.(null)}>
      <Axis x={sx} y={sy} w={width - margin.r} h={height - margin.b}
        xTicks={ticks} yTicks={yt} xLabel={xLabel} yLabel={yLabel}
        xFormat={xFormat} yFormat={yFormat} />
      {series.map((s, k) =>
        s.area ? (
          <path key={`a${k}`} d={`${path(s.values)} L${sx(s.values.length - 1)},${sy(lo)} L${sx(0)},${sy(lo)} Z`}
            fill={s.color} opacity={0.12} />
        ) : null
      )}
      {series.map((s, k) => (
        <path key={k} d={path(s.values)} fill="none" stroke={s.color}
          strokeWidth={s.width ?? 1.6} strokeDasharray={s.dash} opacity={s.opacity ?? 1}
          strokeLinejoin="round" strokeLinecap="round" />
      ))}
      {hoverIndex != null && (
        <line x1={sx(hoverIndex)} x2={sx(hoverIndex)} y1={margin.t} y2={height - margin.b}
          stroke="var(--ink)" strokeWidth={1} opacity={0.35} />
      )}
      {Array.from({ length: n }, (_, i) => (
        <rect key={i} x={sx(i) - (width - margin.l - margin.r) / (2 * n)} y={margin.t}
          width={(width - margin.l - margin.r) / n} height={height - margin.t - margin.b}
          fill="transparent" onMouseEnter={() => onHover?.(i)} />
      ))}
    </svg>
  );
}

export function Heat({
  rows, cols, value, color, cellW = 10, cellH = 10, gap = 1,
  rowLabels, colLabels, onHover, highlight, formatValue,
}: {
  rows: number; cols: number; value: (r: number, c: number) => number;
  color: (v: number) => string; cellW?: number; cellH?: number; gap?: number;
  rowLabels?: (r: number) => string; colLabels?: (c: number) => string;
  onHover?: (r: number, c: number) => void;
  highlight?: { r: number; c: number } | null;
  formatValue?: (v: number) => string;
}) {
  const left = rowLabels ? 56 : 0;
  const top = colLabels ? 16 : 0;
  return (
    <svg width={left + cols * (cellW + gap)} height={top + rows * (cellH + gap)} className="heat">
      {Array.from({ length: rows }, (_, r) =>
        Array.from({ length: cols }, (_, c) => {
          const v = value(r, c);
          const on = highlight && highlight.r === r && highlight.c === c;
          return (
            <rect key={`${r}-${c}`} x={left + c * (cellW + gap)} y={top + r * (cellH + gap)}
              width={cellW} height={cellH} fill={color(v)}
              stroke={on ? "var(--ink)" : "none"} strokeWidth={on ? 1.4 : 0}
              onMouseEnter={() => onHover?.(r, c)}>
              <title>{formatValue ? formatValue(v) : fmt(v, 3)}</title>
            </rect>
          );
        })
      )}
      {rowLabels && Array.from({ length: rows }, (_, r) =>
        r % Math.max(1, Math.floor(rows / 14)) === 0 ? (
          <text key={r} x={left - 5} y={top + r * (cellH + gap) + cellH * 0.7}
            textAnchor="end" className="ticklabel">{rowLabels(r)}</text>
        ) : null
      )}
      {colLabels && Array.from({ length: cols }, (_, c) =>
        c % Math.max(1, Math.floor(cols / 12)) === 0 ? (
          <text key={c} x={left + c * (cellW + gap) + cellW / 2} y={10}
            textAnchor="middle" className="ticklabel">{colLabels(c)}</text>
        ) : null
      )}
    </svg>
  );
}

export function Matrix({
  data, size, color, labels, onHover,
}: {
  data: number[][]; size: number; color: (v: number) => string;
  labels?: string[]; onHover?: (r: number, c: number) => void;
}) {
  const left = labels ? 32 : 0;
  const top = labels ? 16 : 0;
  const cell = size / data.length;
  return (
    <svg width={left + size} height={top + size} className="matrix">
      {data.map((row, r) =>
        row.map((v, c) => (
          <rect key={`${r}-${c}`} x={left + c * cell} y={top + r * cell}
            width={Math.max(0.6, cell - 0.6)} height={Math.max(0.6, cell - 0.6)}
            fill={color(v)} onMouseEnter={() => onHover?.(r, c)}>
            <title>{`${labels?.[r] ?? r} × ${labels?.[c] ?? c}: ${v.toFixed(3)}`}</title>
          </rect>
        ))
      )}
      {labels && labels.map((l, i) =>
        i % Math.max(1, Math.floor(labels.length / 10)) === 0 ? (
          <g key={i}>
            <text x={left - 5} y={top + i * cell + cell * 0.7} textAnchor="end" className="ticklabel">{l}</text>
            <text x={left + i * cell + cell * 0.5} y={10} textAnchor="middle" className="ticklabel">{l}</text>
          </g>
        ) : null
      )}
    </svg>
  );
}

export function HBars({
  items, max, width = 520, labelWidth = 120, height = 18, gap = 5, format,
}: {
  items: { label: string; value: number; color?: string; sub?: string }[];
  max?: number; width?: number; labelWidth?: number; height?: number; gap?: number;
  format?: (v: number) => string;
}) {
  const hi = max ?? Math.max(...items.map((d) => d.value), 1e-9);
  return (
    <div className="hbars">
      {items.map((d, i) => (
        <div className="hbar-row" key={i} style={{ marginBottom: gap }}>
          <span className="hbar-label" style={{ width: labelWidth }}>{d.label}</span>
          <span className="hbar-track" style={{ height }}>
            <span className="hbar-fill" style={{ width: `${(d.value / hi) * 100}%`, background: d.color ?? "var(--signal)" }} />
          </span>
          <span className="hbar-val">{format ? format(d.value) : fmt(d.value, 2)}</span>
          {d.sub && <span className="hbar-sub">{d.sub}</span>}
        </div>
      ))}
    </div>
  );
}

export function Scatter({
  width, height, points, xLabel, yLabel, xDomain, yDomain, xTicks, yTicks,
  xFormat, yFormat, size = 3.4, opacity = 0.72, onHover, highlight,
}: {
  width: number; height: number;
  points: { x: number; y: number; color: string; label?: string; r?: number }[];
  xLabel?: string; yLabel?: string; xDomain?: [number, number]; yDomain?: [number, number];
  xTicks?: number[]; yTicks?: number[]; xFormat?: (v: number) => string; yFormat?: (v: number) => string;
  size?: number; opacity?: number; onHover?: (i: number | null) => void; highlight?: number | null;
}) {
  const m = { t: 14, r: 20, b: 32, l: 48 };
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const [x0, x1] = xDomain ?? [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = yDomain ?? [Math.min(...ys), Math.max(...ys)];
  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * (width - m.l - m.r);
  const sy = (v: number) => height - m.b - ((v - y0) / (y1 - y0 || 1)) * (height - m.t - m.b);
  const xt = xTicks ?? Array.from({ length: 5 }, (_, i) => x0 + (i / 4) * (x1 - x0));
  const yt = yTicks ?? Array.from({ length: 5 }, (_, i) => y0 + (i / 4) * (y1 - y0));
  return (
    <svg width={width} height={height} className="chart">
      <Axis x={sx} y={sy} w={width - m.r} h={height - m.b} xTicks={xt} yTicks={yt}
        xLabel={xLabel} yLabel={yLabel} xFormat={xFormat} yFormat={yFormat} />
      {points.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={p.r ?? (highlight === i ? 6 : size)}
          fill={p.color} opacity={highlight === i ? 1 : opacity}
          stroke={highlight === i ? "var(--ink)" : "none"} strokeWidth={1.2}
          onMouseEnter={() => onHover?.(i)}>
          <title>{p.label ?? `${fmt(p.x, 2)}, ${fmt(p.y, 2)}`}</title>
        </circle>
      ))}
    </svg>
  );
}

export function LayerRuler({
  layers, x, y, w, active, onPick, every = 4,
}: {
  layers: number; x: (l: number) => number; y: number; w: number;
  active?: number; onPick?: (l: number) => void; every?: number;
}) {
  return (
    <g>
      <line x1={x(0)} x2={x(layers - 1)} y1={y} y2={y} stroke="var(--ink)" />
      {Array.from({ length: layers }, (_, l) => (
        <g key={l}>
          <line x1={x(l)} x2={x(l)} y1={y - 3} y2={y + 3}
            stroke={active === l ? "var(--ember)" : "var(--rule-2)"}
            strokeWidth={active === l ? 2 : 1} />
          {l % every === 0 && (
            <text x={x(l)} y={y + 13} textAnchor="middle" className="ticklabel">{l}</text>
          )}
          {onPick && (
            <rect x={x(l) - w / (2 * layers)} y={y - 7} width={w / layers} height={22}
              fill="transparent" style={{ cursor: "pointer" }} onClick={() => onPick(l)} />
          )}
        </g>
      ))}
    </g>
  );
}

export function Sparkline({ values, width = 80, height = 18, color = "var(--signal)" }: {
  values: number[]; width?: number; height?: number; color?: string;
}) {
  const lo = Math.min(...values), hi = Math.max(...values);
  const sx = (i: number) => (i / Math.max(1, values.length - 1)) * width;
  const sy = (v: number) => height - ((v - lo) / (hi - lo || 1)) * height;
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <path d={values.map((v, i) => `${i ? "L" : "M"}${sx(i)},${sy(v)}`).join(" ")}
        fill="none" stroke={color} strokeWidth={1.2} />
    </svg>
  );
}

export function ProbabilityDots({
  probs, target, width = 420,
}: {
  probs: number[]; target: number; width?: number;
}) {
  return (
    <div className="prob-dots" style={{ width }}>
      {probs.map((p, i) => {
        const on = p >= target;
        return (
          <span key={i} className={"prob-dot" + (on ? " on" : "")}
            style={{ opacity: 0.15 + 0.85 * Math.sqrt(p / Math.max(...probs, 1e-9)) }}
            title={`${pct(p, 2)}${on ? " · sampled" : ""}`} />
        );
      })}
    </div>
  );
}
