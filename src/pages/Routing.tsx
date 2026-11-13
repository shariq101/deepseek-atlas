import React, { useMemo, useState } from "react";
import {
  Meta, Prompt, RouterEntry, fmt, pct, tokenLabel, useMeasure, useOne, SEQ, ramp,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart } from "../components/charts";

function expertColor(id: number, n: number) {
  return `hsl(${(id * 137.508) % 360} 52% ${34 + (id % 3) * 7}%)`;
}

export default function Routing() {
  const metaQ = useOne<Meta>("meta");
  const promptQ = useOne<Prompt[]>("prompts");
  const routeQ = useOne<Record<string, RouterEntry>>("router");
  const [pid, setPid] = useState<string | null>(null);
  const [gate, setGate] = useState(12);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const prompts = promptQ.data ?? [];
  const routing = routeQ.data;
  const prompt = prompts.find((p) => p.id === pid) ?? prompts.find((p) => routing?.[p.id]) ?? prompts[0];
  const r = prompt && routing ? routing[prompt.id] : null;
  const L = meta?.layers ?? 27;
  const E = meta?.experts ?? 64;
  const K = meta?.topk ?? 6;
  // The first_k_dense_replace layer(s) have no router. The capture stores the
  // gate-layer ids; older artifacts infer the tail block.
  const gateLayers = useMemo(() => {
    if (!r) return [];
    if (r.layers?.length) return r.layers;
    const offset = L - r.top.length;
    return Array.from({ length: r.top.length }, (_, i) => offset + i);
  }, [r, L]);
  const G = gateLayers.length;

  const stats = useMemo(() => {
    if (!r) return null;
    // Captured entropy is stored with a singleton token axis ([layer][1][token]);
    // flatten before averaging so the chart never receives NaN.
    const ent = r.entropy.map((row) => {
      const vals = row.flat(Infinity) as number[];
      return vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
    });
    const mar = r.margin.map((row) => row.reduce((a, b) => a + b, 0) / row.length);
    const used = new Set<number>();
    for (let l = 0; l < r.top.length; l++) for (const row of r.top[l].slice(0, r.tokens.length)) for (const [e] of row) used.add(e);
    const loadAt = new Array(E).fill(0);
    // top[] includes the generated target position; the grid only labels the
    // prompt positions, so keep load counts on the same basis.
    for (const row of (r.top[gate] ?? []).slice(0, r.tokens.length)) for (const [e] of row) loadAt[e]++;
    return { ent, mar, used: used.size, loadAt };
  }, [r, gate, E]);

  return (
    <>
      <ChapterHead n="09" part="II" kicker="Routing"
        title={<>Six experts, chosen <em>per token</em></>}
        standfirst={<>The router is the model's only discrete decision. At every MoE block it scores 64 experts, keeps six, and weights their outputs. The choices are not random, but they are also not a clean semantic taxonomy — which is exactly what makes them worth measuring.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>{E} routed + 2 shared</Chip><Chip>top-{K}</Chip></>}
      />
      <DataState loading={metaQ.loading || promptQ.loading || routeQ.loading}
        error={metaQ.error ?? promptQ.error ?? routeQ.error} what="the router capture">
        {prompt && r && stats && (() => {
          const cell = Math.max(8, Math.min(22, (width - 110) / Math.max(G, 1)));
          return (
          <>
            <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
              <label className="field">
                <span>Prompt</span>
                <select className="ctl" value={prompt.id} onChange={(e) => setPid(e.target.value)} style={{ minWidth: 340 }}>
                  {prompts.filter((p) => routing?.[p.id]).map((p) => (
                    <option key={p.id} value={p.id}>{p.id} — {p.text.slice(0, 56).replace(/\n/g, " ")}</option>
                  ))}
                </select>
              </label>
              <div style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>{r.tokens.length} tokens × {G} MoE layers</span>
            </div>

            <Panel title={<><b>Figure 9.1</b> · the routing grid</>}
              note="Each row is a token, each column a MoE layer; the cell shows the router's top-1 expert, coloured by expert identity. Vertical stripes mean a layer consistently sends neighbouring tokens to the same expert; horizontal bands mean one token's choice persists across depth.">
              <div style={{ overflowX: "auto" }}>
                <div style={{ display: "inline-block" }}>
                  <div style={{ display: "flex", gap: 1, marginLeft: 108, marginBottom: 3 }}>
                    {Array.from({ length: G }, (_, gi) => (
                      <div key={gi} className="mono" style={{ width: cell, fontSize: 7.5, color: "var(--ink-3)", textAlign: "center" }}>{gi % 4 === 0 ? gateLayers[gi] : ""}</div>
                    ))}
                  </div>
                  {r.tokens.map((tok, t) => (
                    <div key={t} style={{ display: "flex", alignItems: "center", gap: 1, marginBottom: 1 }}>
                      <div className="mono" style={{ width: 104, fontSize: 8.5, color: "var(--ink-2)", textAlign: "right", paddingRight: 5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {tokenLabel(tok) || "∅"}
                      </div>
                      {Array.from({ length: G }, (_, gi) => {
                        const top = r.top[gi]?.[t] ?? [];
                        const e = top[0]?.[0] ?? -1;
                        return (
                          <div key={gi} title={`L${gateLayers[gi]} · ${tokenLabel(tok)} · ${top.map(([id, w]) => `E${id} ${pct(w, 0)}`).join(" · ")}`}
                            style={{ width: cell, height: 15, background: e < 0 ? "transparent" : expertColor(e, E), opacity: 0.9 }} />
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
              <div className="caption">Colour is expert identity, not weight. Hover a cell for the full top-{K} and its gate weights.</div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 9.2</b> · load at one layer</>}
                tools={<span className="mono" style={{ fontSize: 9 }}>L{gateLayers[gate] ?? "—"}</span>}
                note="How many tokens selected each expert at the chosen MoE layer. A balanced router gives every expert roughly n·k/64 selections; spikes are routing collapse, and empty bars are experts this prompt never uses.">
                <input type="range" min={0} max={Math.max(G - 1, 0)} value={gate} onChange={(e) => setGate(+e.target.value)} style={{ marginBottom: 8 }} />
                <svg width={Math.max(width / 2, 320)} height={150} className="chart">
                  {stats.loadAt.map((v, e) => {
                    const max = Math.max(...stats.loadAt, 1);
                    const w = Math.max(width / 2, 320) / E;
                    return <rect key={e} x={e * w} y={140 - (v / max) * 120} width={w - 1} height={(v / max) * 120}
                      fill={expertColor(e, E)} opacity={0.9}><title>{`E${e}: ${v} tokens`}</title></rect>;
                  })}
                  <line x1={0} x2={Math.max(width / 2, 320)} y1={140} y2={140} stroke="var(--ink)" />
                </svg>
                <div className="caption">64 experts, 0–{Math.max(...stats.loadAt)} selections per expert.</div>
              </Panel>

              <Panel title={<><b>Figure 9.3</b> · decision confidence</>}
                note="Mean gate entropy over all 64 softmax logits (range 0–4.16 nats), and the selected top-1 minus top-2 weight, per MoE layer. Low entropy means the router is decisive; neither curve is monotonic with depth here.">
                <div ref={ref}>
                  <LineChart width={Math.max(width / 2 - 30, 320)} height={190}
                    series={[
                      { values: stats.ent, color: "var(--signal)", label: "entropy", width: 1.8 },
                      { values: stats.mar, color: "var(--ember)", label: "margin", width: 1.8 },
                    ]}
                    xTicks={[0, 7, 14, 20, Math.max(G - 1, 0)]} xFormat={(i) => String(gateLayers[Math.round(i)] ?? i)} xLabel="MoE layer" yFormat={(v) => fmt(v, 2)} />
                </div>
                <div className="legend" style={{ marginTop: 6 }}>
                  <Legend items={[{ c: "var(--signal)", label: "entropy (nats)", line: true }, { c: "var(--ember)", label: "top-1 − top-2", line: true }]} />
                </div>
              </Panel>
            </div>

            <div className="grid4">
              <Stat k="Experts used" v={`${stats.used} / ${E}`} s="by this prompt" />
              <Stat k="Mean entropy" v={fmt(stats.ent.reduce((a, b) => a + b, 0) / Math.max(G, 1), 2)} s="nats" />
              <Stat k="Mean margin" v={fmt(stats.mar.reduce((a, b) => a + b, 0) / Math.max(G, 1), 3)} s="weight" />
              <Stat k="Top-1 share" v={pct((r.top[gate] ?? []).slice(0, r.tokens.length).filter((row) => row[0]?.[0] === 0).length / Math.max(r.tokens.length, 1), 0)} s="expert 0" />
            </div>

            <h2 className="sec">The router is a learned index, not a topic model</h2>
            <Prose>
              <p>
                The grid rarely shows clean semantic columns. What it shows is{" "}
                <em>token-level specialisation</em>: punctuation and function words route
                consistently across layers, while content tokens change experts as the
                residual moves from surface form to answer. The same expert can serve
                “Paris”, a bracket, and a digit in different contexts; that is not a bug,
                it is what a low-rank gate with 64 cells can represent.
              </p>
              <p>
                Load is the quantity that matters for throughput. If one expert receives
                three times its share, the MoE layer waits on it and the nominal active
                parameter count overstates speed. DeepSeek trains an auxiliary
                load-balancing loss and uses a shared expert to absorb common computation, but
                neither guarantees uniform load at inference on a single prompt.
              </p>
            </Prose>

            <Method>
              <p>
                For every MoE block, the gate's top-{K} indices and softmax weights are
                captured after the routing softmax. Load is the count of top-{K}
                selections per expert; entropy is over all 64 gate probabilities. Margin
                is the top-1 minus top-2 selected weight.
              </p>
              <p>
                The shared experts are excluded from the grid: they are always active and
                carry no routing decision. Their contribution is measured in chapter 13.
              </p>
            </Method>

            <Footnotes items={[
              `Router hooks on ${meta?.modelName ?? "V2-Lite"} ${meta?.precision ?? "NF4"}; gate weights are bf16. ${meta?.corpus ?? 27} prompts, deep grid for ${Object.keys(routing ?? {}).length}.`,
              "Expert identity is the index in the checkpoint, not a label. The colour map is arbitrary and stable across prompts.",
              "Entropy is computed over all 64 gate probabilities. A router can have low entropy and still be wrong.",
              "Load balance within one prompt is not the training objective; it is an inference-time observation.",
            ]} />
          </>
          );
        })()}
      </DataState>
      <PageFoot index={9} />
    </>
  );
}
