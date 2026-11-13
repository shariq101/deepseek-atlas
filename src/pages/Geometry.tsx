import React, { useMemo, useState } from "react";
import { Geometry, Meta, fmt, pct, tokenLabel, useMeasure, useOne, SEQ, ramp } from "../lib/atlas";
import {
  ChapterHead, Chip, Colorbar, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart, Scatter } from "../components/charts";

const COLORS = ["var(--signal)", "var(--ember)", "var(--teal)", "var(--olive)", "var(--plum)", "var(--sand)"];

type Pt = [number, number];

function quantileSorted(values: number[], q: number): number {
  const pos = (values.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const a = values[base] ?? values[values.length - 1] ?? 0;
  const b = values[base + 1] ?? a;
  return a + rest * (b - a);
}

/** Tukey-fence view range. The data itself is never clipped or rewritten. */
function robustDomain(values: number[], pad = 0.08): [number, number] {
  const xs = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!xs.length) return [-1, 1];
  const first = xs[0] ?? 0;
  const last = xs[xs.length - 1] ?? first;
  if (xs.length < 4) {
    const span = last - first || Math.max(1, Math.abs(last) * 0.1);
    return [first - span * pad, last + span * pad];
  }
  const q1 = quantileSorted(xs, 0.25);
  const q3 = quantileSorted(xs, 0.75);
  const iqr = q3 - q1;
  const lo = iqr > 0 ? q1 - 1.5 * iqr : first;
  const hi = iqr > 0 ? q3 + 1.5 * iqr : last;
  const span = hi - lo || Math.max(1, Math.abs(hi) * 0.1);
  return [lo - span * pad, hi + span * pad];
}

function quantileDomain(values: number[], low = 0.02, high = 0.98, pad = 0.08): [number, number] {
  const xs = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!xs.length) return [-1, 1];
  const first = xs[0] ?? 0;
  const last = xs[xs.length - 1] ?? first;
  if (xs.length < 4) {
    const span = last - first || Math.max(1, Math.abs(last) * 0.1);
    return [first - span * pad, last + span * pad];
  }
  const qLo = quantileSorted(xs, low);
  const qHi = quantileSorted(xs, high);
  const span = qHi - qLo || Math.max(1, Math.abs(qHi) * 0.1);
  return [qLo - span * pad, qHi + span * pad];
}

function outside(p: Pt, xDomain: [number, number], yDomain: [number, number]): boolean {
  return p[0] < xDomain[0] || p[0] > xDomain[1] || p[1] < yDomain[0] || p[1] > yDomain[1];
}

function ticks(lo: number, hi: number, count = 5): number[] {
  return Array.from({ length: count }, (_, i) => lo + (i / (count - 1)) * (hi - lo));
}

export default function GeometryPage() {
  const metaQ = useOne<Meta>("meta");
  const geoQ = useOne<Geometry>("geometry");
  const [layer, setLayer] = useState(18);
  const [tid, setTid] = useState<string | null>(null);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const rank = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const geo = geoQ.data;
  const L = geo?.cka.length ?? meta?.layers ?? 27;
  const tids = geo ? Object.keys(geo.tokenTrajectories) : [];
  const traj = geo?.tokenTrajectories[tid ?? tids[0]];

  const driftPeak = useMemo(() => {
    if (!geo) return 0;
    return geo.drift.indexOf(Math.max(...geo.drift));
  }, [geo]);

  return (
    <>
      <ChapterHead n="15" part="II" kicker="Representation geometry"
        title={<>How the space <em>moves</em></>}
        standfirst={<>If layers were a stack of independent functions, the representation would jump at every block. It does not. CKA, effective rank and drift measure how much the residual space is preserved, and where the model actually reorganises.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>{L} layers</Chip><Chip>linear CKA</Chip></>}
      />
      <DataState loading={metaQ.loading || geoQ.loading} error={metaQ.error ?? geoQ.error} what="the geometry capture">
        {geo && (
          <>
            <div className="grid2">
              <Panel title={<><b>Figure 15.1</b> · similarity between every pair of layers</>}
                note="Linear CKA across the 27 prompt-level residual states. The bright diagonal is trivial; the question is how wide the bright band is. A wide band means many layers compute in a shared space; a narrow one means each block rotates it.">
                <Heat rows={L} cols={L} cellW={width > 700 ? 11 : 9} cellH={width > 700 ? 11 : 9} gap={0.5}
                  value={(a, b) => geo.cka[a]?.[b] ?? 0}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(l) => (l % 4 === 0 ? String(l) : "")}
                  colLabels={(l) => (l % 4 === 0 ? String(l) : "")}
                  formatValue={(v) => fmt(v, 2)} />
                <div style={{ marginTop: 6 }}>
                  <Colorbar stops={SEQ} min={0} max={1} label="linear CKA" format={(v) => fmt(v, 2)} />
                </div>
              </Panel>

              <Panel title={<><b>Figure 15.2</b> · rank and drift</>}
                note="Effective rank of the 27 prompt-level residual states, and the cosine drift of the final prompt's final-position state from layer to layer. With n=27, rank is capped at 27 and is descriptive; the curve is not monotonic. Drift peaks where that state changes direction fastest.">
                <div ref={rank.ref}>
                  <LineChart width={Math.max(rank.width, 300)} height={210}
                    series={[{ values: geo.effectiveRank, color: "var(--ink)", label: "effective rank", width: 2 }]}
                    xTicks={[0, 7, 14, 20, L - 1]} xLabel="layer" yFormat={(v) => fmt(v, 0)} />
                </div>
                <div style={{ marginTop: 12 }}>
                  <LineChart width={Math.max(rank.width, 300)} height={150}
                    series={[{ values: geo.drift, color: "var(--ember)", label: "cosine drift", width: 1.6 }]}
                    xTicks={[0, 7, 14, 20, Math.max(geo.drift.length - 1, 0)]} xLabel="layer" yFormat={(v) => fmt(v, 2)} />
                </div>
              </Panel>
            </div>

            <Panel title={<><b>Figure 15.3</b> · the cloud, coloured by prompt</>}
              tools={<span className="mono" style={{ fontSize: 9 }}>L{layer}</span>}
              note="PCA of the final-position residual at one depth, one point per prompt. Colour cycles with prompt identity. Tight clusters mean prompts share a representation at this depth; separated clusters mean prompt identity still dominates. Axes use robust Tukey fences; off-scale prompts are named below the plot.">
              <input type="range" min={0} max={L - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} />
              <div ref={ref}>
                {(() => {
                  const points = (geo.pca.points[layer] ?? []).map((p) => [p[0], p[1]] as Pt);
                  const [x0, x1] = robustDomain(points.map((p) => p[0]));
                  const [y0, y1] = robustDomain(points.map((p) => p[1]));
                  const indexed = points.map((p, i) => ({ p, i }));
                  const offScale = indexed.filter(({ p }) => outside(p, [x0, x1], [y0, y1]));
                  const visible = indexed.filter(({ p }) => !outside(p, [x0, x1], [y0, y1]));
                  return (
                    <>
                      <Scatter width={Math.max(width, 380)} height={340}
                        points={visible.map(({ p, i }) => ({
                          x: p[0], y: p[1], r: 3.2,
                          color: COLORS[i % COLORS.length],
                          label: `${geo.promptIds[i] ?? `point ${i}`} · PC1 ${fmt(p[0], 2)} · PC2 ${fmt(p[1], 2)}`,
                        }))}
                        xLabel={`PC1 · ${pct(geo.pca.explained[0] ?? 0, 0)}`}
                        yLabel={`PC2 · ${pct(geo.pca.explained[1] ?? 0, 0)}`}
                        xDomain={[x0, x1]} yDomain={[y0, y1]}
                        xFormat={(v) => fmt(v, 2)} yFormat={(v) => fmt(v, 2)}
                        opacity={0.9} />
                      <div className="caption">
                        {offScale.length > 0
                          ? <>Robust view range. Off-scale at this depth: <b>{offScale.map(({ p, i }) => `${geo.promptIds[i] ?? `point ${i}`} (PC1 ${fmt(p[0], 1)}, PC2 ${fmt(p[1], 1)})`).join(", ")}</b>. These points are not drawn in this view; the capture is unchanged.</>
                          : "Robust view range. Every prompt is inside the current view."}
                      </div>
                    </>
                  );
                })()}
              </div>
            </Panel>

            <Panel title={<><b>Figure 15.4</b> · token trajectories through depth</>}
              note="Each line follows one token position across layers in the shared PCA basis. A trajectory that moves steadily is a representation being refined; a sharp turn is a block changing what the token means in context. Axes use robust view ranges; off-scale trajectory segments are clipped and named in the plot and caption.">
              <div className="row" style={{ marginBottom: 6 }}>
                <select className="ctl" value={tid ?? tids[0]} onChange={(e) => setTid(e.target.value)} style={{ minWidth: 300 }}>
                  {tids.map((id) => <option key={id} value={id}>{id}</option>)}
                </select>
              </div>
              {traj && (() => {
                const W = Math.max(width, 500);
                const H = 320;
                const m = { t: 18, r: 24, b: 38, l: 52 };
                const pw = W - m.l - m.r;
                const ph = H - m.t - m.b;
                const all = traj.points.flat().map((p) => [p[0], p[1]] as Pt);
                const [x0, x1] = robustDomain(all.map((p) => p[0]), 0.12);
                const [y0, y1] = quantileDomain(all.map((p) => p[1]), 0.02, 0.98, 0.12);
                const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * pw;
                const sy = (v: number) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * ph;
                const nTokens = traj.points[0]?.length ?? 0;
                const tokenIds = Array.from({ length: nTokens }, (_, t) => t);
                const tokenName = (t: number) => tokenLabel(traj.tokens[t] ?? "").replace(/\s+/g, " ").trim();
                const label = (t: number) => tokenName(t) || `token ${t}`;
                const offLabel = (t: number) => `token ${t}${tokenName(t) ? ` "${tokenName(t)}"` : ""}`;
                const offScale = tokenIds.filter((t) => traj.points.some((layerPts) => {
                  const p = layerPts[t];
                  return p ? outside([p[0], p[1]], [x0, x1], [y0, y1]) : false;
                }));
                const xTicks = ticks(x0, x1);
                const yTicks = ticks(y0, y1);
                return (
                  <div style={{ overflowX: "auto" }}>
                    <svg width={W} height={H} className="chart">
                      <defs>
                        <clipPath id="trajectory-clip">
                          <rect x={m.l} y={m.t} width={pw} height={ph} />
                        </clipPath>
                      </defs>
                      <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="var(--rule-2)" />
                      <line x1={m.l} x2={m.l} y1={m.t} y2={H - m.b} stroke="var(--rule-2)" />
                      {xTicks.map((v, i) => (
                        <g key={`x-${i}`}>
                          <line x1={sx(v)} x2={sx(v)} y1={H - m.b} y2={H - m.b + 4} stroke="var(--ink)" />
                          <text x={sx(v)} y={H - m.b + 16} textAnchor="middle" className="ticklabel">{fmt(v, 0)}</text>
                        </g>
                      ))}
                      {yTicks.map((v, i) => (
                        <g key={`y-${i}`}>
                          <line x1={m.l - 4} x2={m.l} y1={sy(v)} y2={sy(v)} stroke="var(--ink)" />
                          <text x={m.l - 8} y={sy(v) + 3} textAnchor="end" className="ticklabel">{fmt(v, 0)}</text>
                        </g>
                      ))}
                      <text x={m.l + pw / 2} y={H - 5} textAnchor="middle" className="ticklabel">PC1</text>
                      <text transform={`translate(14 ${m.t + ph / 2}) rotate(-90)`} textAnchor="middle" className="ticklabel">PC2</text>
                      <g clipPath="url(#trajectory-clip)">
                        {tokenIds.map((t) => (
                          <polyline key={t} fill="none" stroke={`hsl(${(t * 37) % 360} 52% 38%)`} strokeWidth={1.35} opacity={0.86}
                            points={traj.points.map((layerPts) => layerPts[t]).filter(Boolean).map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")}>
                            <title>{label(t)}</title>
                          </polyline>
                        ))}
                      </g>
                      {offScale.length > 0 && (
                        <text x={m.l + 4} y={m.t + 12} textAnchor="start" className="ticklabel" fill="var(--ember)">
                          off-scale: {offScale.length > 3
                            ? `${offScale.slice(0, 3).map(offLabel).join(", ")} +${offScale.length - 3} more`
                            : offScale.map(offLabel).join(", ")}
                        </text>
                      )}
                    </svg>
                    <div className="caption">
                      One line per token position; the last point is the final layer. Robust view range; axes show PC1 and PC2.
                      {offScale.length > 0
                        ? <> Off-scale trajectory segments are clipped: <b>{offScale.map(offLabel).join(", ")}</b>. Their residual states are real outliers, not missing data; the capture is unchanged.</>
                        : " Every token trajectory is inside the current view."}
                    </div>
                  </div>
                );
              })()}
            </Panel>

            <div className="grid4">
              <Stat k="Effective rank" v={fmt(geo.effectiveRank[Math.floor(L / 2)] ?? 0, 0)} s={`at L${Math.floor(L / 2)}`} />
              <Stat k="Peak drift" v={`L${driftPeak}`} s={fmt(geo.drift[driftPeak] ?? 0, 2)} />
              <Stat k="CKA L0→Llast" v={fmt(geo.cka[0]?.[L - 1] ?? 0, 2)} s="linear similarity" />
              <Stat k="Prompts" v={geo.promptIds.length} s="geometry basis" />
            </div>

            <Prose>
              <p>
                The CKA map is not a smooth diagonal: 19 of 27 adjacent pairs exceed
                0.9, but abrupt drops at L4 and L24 break the band, and L0→Llast is only
                0.006. The model often edits within a shared space, then reorganises
                sharply at a few boundaries. That is consistent with a residual stream,
                but it is not a claim that every block preserves the representation.
              </p>
              <p>
                The exceptions are the interesting part. The drift peak at L24 is one
                such boundary; this page does not establish that effective rank and the
                logit lens peak at the same layer. Geometry, computation and readout are
                separate measurements, which is why all three are shown.
              </p>
            </Prose>

            <Method>
              <p>
                Residual states are captured at the final position of every prompt (and all
                positions for the deep prompts). CKA is linear CKA from normalised Gram
                matrices at each layer pair. Effective rank is the exponential of the
                spectral entropy of the covariance. Drift is
                <code>1 − cos(h_l, h_l+1)</code> for the last prompt's final-position state.
              </p>
              <p>
                PCA is fitted once on the pooled states of all layers so that trajectories
                are comparable across depth; the explained-variance percentages are for that
                global basis, not per layer. The global basis is sensitive to extreme states:
                the first token position has a residual-norm excursion near 2.9k from L4,
                and <code>arith-mul</code> is an extreme late-layer outlier. Figures 15.3 and
                15.4 use robust view ranges and name every off-scale point or trajectory;
                those values stay in the capture and are clipped or omitted only from the drawn view.
              </p>
            </Method>

            <Footnotes items={[
              `Measured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); ${geo.promptIds.length} prompts.`,
              "CKA is invariant to invertible linear transforms, so a high value does not mean the layers compute the same function.",
              "Effective rank is sensitive to the prompt distribution and to centring; treat it as a relative curve.",
              "The PCA basis is global; a trajectory can look straight while the layer-local geometry rotates.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={15} />
    </>
  );
}
