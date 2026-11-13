import React, { useMemo, useState } from "react";
import {
  DlaEntry, Meta, NormsEntry, Prompt, fmt, pct, tokenLabel, useMeasure, useOne, SEQ, ramp,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart } from "../components/charts";

export default function Anatomy() {
  const metaQ = useOne<Meta>("meta");
  const promptQ = useOne<Prompt[]>("prompts");
  const normsQ = useOne<Record<string, NormsEntry>>("norms");
  const dlaQ = useOne<Record<string, DlaEntry>>("dla");
  const [pid, setPid] = useState<string | null>(null);
  const [view, setView] = useState<"update" | "dla">("update");
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const prompts = promptQ.data ?? [];
  const norms = normsQ.data;
  const dla = dlaQ.data;
  const prompt = prompts.find((p) => p.id === pid) ?? prompts.find((p) => norms?.[p.id]) ?? prompts[0];
  const n = prompt && norms ? norms[prompt.id] : null;
  const d = prompt && dla ? dla[prompt.id] : null;
  const L = meta?.layers ?? 27;

  const posPrompt = useMemo(
    () => prompts.find((p) => norms?.[p.id]?.posNorm?.length) ?? null,
    [prompts, norms],
  );
  const pos = posPrompt && norms ? norms[posPrompt.id].posNorm! : null;

  return (
    <>
      <ChapterHead n="08" part="II" kicker="Block anatomy"
        title={<>What each block <em>adds</em></>}
        standfirst={<>A transformer is a residual stream with 27 reads and writes. Measuring the size of each write, its direction, and its direct effect on the target logit turns the block diagram into a budget: where the computation lives, and how much of it is attention.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>{L} blocks</Chip><Chip>RMSNorm · SwiGLU</Chip></>}
      />
      <DataState loading={metaQ.loading || promptQ.loading || normsQ.loading || dlaQ.loading}
        error={metaQ.error ?? promptQ.error ?? normsQ.error ?? dlaQ.error} what="the block captures">
        {prompt && n && (
          <>
            <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
              <label className="field">
                <span>Prompt</span>
                <select className="ctl" value={prompt.id} onChange={(e) => setPid(e.target.value)} style={{ minWidth: 340 }}>
                  {prompts.filter((p) => norms?.[p.id]).map((p) => (
                    <option key={p.id} value={p.id}>{p.id} — {p.text.slice(0, 56).replace(/\n/g, " ")}</option>
                  ))}
                </select>
              </label>
              <div style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>{n.tokens.length} positions</span>
            </div>

            <div className="grid2">
              <Panel title={<><b>Figure 8.1a</b> · residual norm by depth</>}
                note="The L2 norm of the residual stream at the final position, before each block. It grows as blocks write into the stream; a flat region is a block that adds little.">
                <div ref={ref}>
                  <LineChart width={Math.max(width / 2 - 30, 320)} height={220}
                    series={[{ values: n.norm, color: "var(--ink)", label: "‖resid‖", width: 2 }]}
                    xTicks={[0, 7, 14, 20, L]} xLabel="layer" yLabel="L2 norm"
                    yFormat={(v) => fmt(v, 0)} />
                </div>
              </Panel>
              <Panel title={<><b>Figure 8.1b</b> · the size of each block update</>}
                note="‖Δresid‖ after each block. The total update is what the residual stream actually receives; the DLA figure below separately projects the attention and MLP parts onto the target logit.">
                <div ref={ref}>
                  <LineChart width={Math.max(width / 2 - 30, 320)} height={220}
                    series={[
                      { values: n.delta, color: "var(--ink-3)", label: "block total", width: 1.8 },
                    ]}
                    xTicks={[0, 7, 14, 20, Math.max(L - 1, 0)]} xLabel="layer" yLabel="‖Δ‖"
                    yFormat={(v) => fmt(v, 0)} />
                </div>
                <div className="legend" style={{ marginTop: 6 }}>
                  <Legend items={[{ c: "var(--ink-3)", label: "block total", line: true }]} />
                </div>
              </Panel>
            </div>

            <Panel title={<><b>Figure 8.2</b> · direct logit attribution</>}
              tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>target: {tokenLabel(d?.targetStr ?? "") || "—"}</span>}
              note="Each block's attention and MLP outputs are projected through the final norm and unembedding onto the target token's logit. Bars are signed contributions; the line is the cumulative residual attribution. This is a decomposition of the readout, not of causality.">
              {d ? (
                <div ref={ref}>
                  <svg width={Math.max(width, 420)} height={280} className="chart">
                    {(() => {
                      const W = Math.max(width, 420), pad = 34, bw = (W - pad * 2) / (L + 1);
                      const vals = [...d.attn, ...d.mlp, ...d.resid];
                      const hi = Math.max(...vals, 0.01), lo = Math.min(...vals, -0.01);
                      const y = (v: number) => 220 - ((v - lo) / (hi - lo)) * 180;
                      return (
                        <>
                          <line x1={pad} x2={W - pad} y1={y(0)} y2={y(0)} stroke="var(--ink)" />
                          {Array.from({ length: L }, (_, l) => (
                            <g key={l}>
                              <rect x={pad + l * bw + 2} y={y(Math.max(0, d.attn[l]))} width={bw * 0.36}
                                height={Math.abs(y(d.attn[l]) - y(0))} fill="var(--signal)" opacity={0.85}>
                                <title>{`L${l} attention ${fmt(d.attn[l], 3)}`}</title>
                              </rect>
                              <rect x={pad + l * bw + bw * 0.4} y={y(Math.max(0, d.mlp[l]))} width={bw * 0.36}
                                height={Math.abs(y(d.mlp[l]) - y(0))} fill="var(--ember)" opacity={0.85}>
                                <title>{`L${l} MLP ${fmt(d.mlp[l], 3)}`}</title>
                              </rect>
                            </g>
                          ))}
                          <polyline fill="none" stroke="var(--ink)" strokeWidth={1.6}
                            points={d.resid.map((v, l) => `${pad + l * bw + bw / 2},${y(v)}`).join(" ")} />
                          <text x={pad} y={246} className="ticklabel">layer 0</text>
                          <text x={W - pad} y={246} textAnchor="end" className="ticklabel">layer {L - 1}</text>
                        </>
                      );
                    })()}
                  </svg>
                  <div className="legend">
                    <Legend items={[{ c: "var(--signal)", label: "attention → target" }, { c: "var(--ember)", label: "MLP → target" }, { c: "var(--ink)", label: "cumulative", line: true }]} />
                  </div>
                </div>
              ) : <div className="caption">No DLA capture for this prompt.</div>}
            </Panel>

            <h2 className="sec">A block is not a uniform object</h2>
            <Prose>
              <p>
                The block update grows sharply with depth, while the residual norm is not
                monotonic: it rises to a mid-stack peak, then falls as later blocks write
                in directions that do not simply add length. The direct logit attribution
                below separates each block's readout effect into attention and MLP terms;
                those are signed projections, not write norms, so the two figures answer
                different questions.
              </p>
              <p>
                Residual norms are not a law of depth. DeepSeek's pre-norm blocks can shrink
                a direction while changing the norm; the cosine between consecutive states
                is the better measure of whether a block preserves or replaces the
                representation. The update norm alone therefore says how much was written,
                not whether the write points toward the answer.
              </p>
            </Prose>

            {pos && (
              <Panel title={<><b>Figure 8.3</b> · the residual across positions</>}
                note={`Per-position residual norm for ${posPrompt?.id}. Rows are token positions, columns layers. The last position usually grows fastest because it accumulates the readout-relevant content.`}>
                <div style={{ overflowX: "auto" }}>
                  <Heat rows={Math.min(pos.length, 22)} cols={L + 1} cellW={19} cellH={13} gap={1}
                    value={(r, c) => {
                      const row = pos[r] ?? [];
                      const hi = Math.max(...row, 1);
                      return (row[c] ?? 0) / hi;
                    }}
                    color={(v) => ramp(v, SEQ)}
                    rowLabels={(r) => (tokenLabel(posPrompt?.text.split(/\s+/)[r] ?? "") || `p${r}`).slice(0, 9)}
                    colLabels={(c) => (c % 5 === 0 ? String(c) : "")}
                    formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
                </div>
              </Panel>
            )}

            <Method>
              <p>
                The residual norm is measured at the final prompt position before each
                block. <code>attn</code> and <code>mlp</code> are the L2 norms of each
                sublayer's output after its residual addition; <code>delta</code> is the
                norm of the block's total update. All are captured with forward hooks, not
                reconstructed.
              </p>
              <p>
                DLA projects each sublayer output through the final RMSNorm and the
                unembedding row for the target token, then sums the projections within
                the sublayer. It is an additive decomposition of the logit only because
                the final readout is linear given the norm; it does not capture
                interactions between blocks.
              </p>
            </Method>

            <Footnotes items={[
              `Measured with ${meta?.precision ?? "NF4"} weights; hook outputs are bf16. ${meta?.corpus ?? 27} prompts available, deep captures for ${Object.keys(norms ?? {}).length}.`,
              "Norm growth is dominated by a few high-norm dimensions; a scalar norm hides direction. Chapter 11 measures the directions.",
              "DLA values are in logits, not probabilities. A +0.3 contribution is meaningful only relative to the target's final logit.",
              "The positional heatmap is row-normalised, so it shows the shape of growth within a position, not absolute magnitude across positions.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={8} />
    </>
  );
}
