import React, { useMemo, useState } from "react";
import {
  GraphEntry, Meta, fmt, tokenLabel, useMeasure, useOne,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";

export default function Circuit() {
  const metaQ = useOne<Meta>("meta");
  const graphQ = useOne<Record<string, GraphEntry>>("graph");
  const [pid, setPid] = useState<string | null>(null);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const graph = graphQ.data;
  const ids = graph ? Object.keys(graph) : [];
  const entry = graph ? graph[pid ?? ids[0]] : null;
  const W = Math.max(width, 640);

  const model = useMemo(() => {
    if (!entry) return null;
    const comps = entry.nodes.filter((n) => n.kind !== "token");
    const byLayer = new Map<number, typeof comps>();
    for (const c of comps) {
      if (!byLayer.has(c.layer)) byLayer.set(c.layer, []);
      byLayer.get(c.layer)!.push(c);
    }
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    const x = (l: number) => 120 + (l / Math.max(...layers, 1)) * (W - 180);
    const pos: Record<string, { x: number; y: number }> = {};
    // One row per prompt position. The graph's token nodes are per-layer top
    // sources; alias them onto these unique rows.
    const tokens = entry.tokens.map((token, i) => ({ id: `tok-${i}`, token, pos: i }));
    const step = Math.min(20, 460 / Math.max(tokens.length, 1));
    tokens.forEach((t, i) => {
      pos[t.id] = { x: 66, y: 42 + i * step };
    });
    for (const node of entry.nodes) {
      if (node.kind === "token") pos[node.id] = pos[`tok-${node.pos}`];
    }
    for (const l of layers) {
      const list = byLayer.get(l)!;
      list.forEach((c, i) => {
        pos[c.id] = { x: x(l), y: 70 + i * 44 };
      });
    }
    const height = Math.max(190, 72 + tokens.length * step);
    return { layers, pos, tokens, comps, height };
  }, [entry, W]);

  const maxW = entry ? Math.max(...entry.edges.map((e) => Math.abs(e.w)), 1e-6) : 1;
  const strongest = entry ? [
    ...entry.attr.attn.map((w, l) => ({ label: `L${l} attention`, w })),
    ...entry.attr.mlp.map((w, l) => ({ label: `L${l} MLP`, w })),
  ].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 5) : [];

  return (
    <>
      <ChapterHead n="14" part="II" kicker="Circuit"
        title={<>The wiring behind a <em>prediction</em></>}
        standfirst={<>A circuit is a claim that a small set of components carries a specific behaviour. This capture supports a smaller claim: it records which prompt positions feed each attention block, and each attention block feeding the MLP that follows. Direct attributions to the target logit are stored separately; they are not the wiring edges.</>}
        chips={<><Chip tone="teal">measured · attribution</Chip><Chip>not a full circuit</Chip><Chip>{ids.length} prompts</Chip></>}
      />
      <DataState loading={graphQ.loading} error={graphQ.error} what="the attribution graph">
        {entry && model && (
          <>
            <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
              <label className="field">
                <span>Prompt</span>
                <select className="ctl" value={pid ?? ids[0]} onChange={(e) => setPid(e.target.value)} style={{ minWidth: 340 }}>
                  {ids.map((id) => <option key={id} value={id}>{id} — {graph?.[id].tokens.join(" ").slice(0, 50)}</option>)}
                </select>
              </label>
              <div style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>target: {tokenLabel(entry.target) || "—"}</span>
            </div>

            <Panel title={<><b>Figure 14.1</b> · token-to-component wiring</>}
              note="Left: one row per prompt token position. Right: attention and MLP blocks, layer by layer. A token→attention edge is a top-three mean-attention source for that layer; the same token row can feed several layers. The chain is attention→MLP and MLP→next attention. This is forward wiring, not a sparsified attribution graph; direct attribution to the target is a separate array plotted in Figure 14.2.">
              <div ref={ref} style={{ overflowX: "auto" }}>
                <svg width={W} height={model.height} style={{ minWidth: 600 }}>
                  <defs>
                    <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
                      <path d="M0,0 L8,4 L0,8 z" fill="var(--rule-2)" />
                    </marker>
                  </defs>
                  {entry.edges.map((e, i) => {
                    const s = model.pos[e.s], t = model.pos[e.t];
                    if (!s || !t) return null;
                    const mx = (s.x + t.x) / 2;
                    return (
                      <path key={i} d={`M${s.x},${s.y} C${mx},${s.y} ${mx},${t.y} ${t.x},${t.y}`}
                        fill="none" stroke={e.w >= 0 ? "var(--signal)" : "var(--ember)"}
                        strokeWidth={0.5 + (Math.abs(e.w) / maxW) * 2.4} opacity={0.12 + (Math.abs(e.w) / maxW) * 0.55}
                        markerEnd="url(#arrow)">
                        <title>{`${e.s} → ${e.t} · ${fmt(e.w, 3)}`}</title>
                      </path>
                    );
                  })}
                  {model.tokens.map((t) => {
                    const p = model.pos[t.id];
                    return (
                      <g key={t.id}>
                        <rect x={p.x - 46} y={p.y - 8} width={92} height={16} rx={2} fill="var(--paper-2)" stroke="var(--rule)" />
                        <text x={p.x} y={p.y + 4} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ink-2)" }}>
                          {(tokenLabel(t.token ?? "") || "∅").slice(0, 12)}
                        </text>
                      </g>
                    );
                  })}
                  {model.comps.map((c) => {
                    const p = model.pos[c.id];
                    return (
                      <g key={c.id}>
                        <circle cx={p.x} cy={p.y} r={6 + Math.min(8, Math.abs(c.attr) * 30)}
                          fill={c.kind === "attn" ? "var(--signal)" : "var(--ember)"} opacity={0.82}>
                          <title>{`${c.kind} L${c.layer} · attribution ${fmt(c.attr, 3)}`}</title>
                        </circle>
                        {c.layer % 4 === 0 && (
                          <text x={p.x} y={22} textAnchor="middle" className="ticklabel">{c.layer}</text>
                        )}
                      </g>
                    );
                  })}
                  <text x={60} y={18} textAnchor="middle" className="ticklabel">tokens</text>
                  <text x={W - 24} y={18} textAnchor="end" className="ticklabel">blocks</text>
                </svg>
              </div>
              <div className="legend" style={{ marginTop: 6, flexWrap: "wrap", gap: 12 }}>
                <span className="legend-item"><span className="swatch" style={{ background: "var(--signal)", borderRadius: "50%" }} /> attention write</span>
                <span className="legend-item"><span className="swatch" style={{ background: "var(--ember)", borderRadius: "50%" }} /> MLP write</span>
                <span className="legend-item" style={{ gap: 6 }}>
                  <span aria-hidden="true" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <i style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--ink-3)" }} />
                    <i style={{ width: 14, height: 14, borderRadius: "50%", background: "var(--ink-3)" }} />
                  </span>
                  component circle size = |direct attribution|, not an abstraction level
                </span>
                <span className="legend-item">left rows = token positions</span>
                <span className="legend-item">rows a / m = attention / MLP write</span>
              </div>
              <div className="caption">Edges are forward wiring, not target contributions. A larger component circle means a larger absolute direct attribution to the target; it does not mean a higher-level abstraction. Token rows are positions, not sized quantities. Edge weights are attention or chain weights. This is a wiring diagram, not a causal circuit.{width < 620 ? " Scroll the strip horizontally to reach all 27 layers." : ""}</div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 14.2</b> · strongest direct attributions</>}
                note="The five largest |attention or MLP → target logit| projections for this prompt. A circuit claim would require ablating these components and showing the behaviour breaks; this table is the hypothesis, not the proof.">
                <div className="stack" style={{ gap: 8 }}>
                  {strongest.map((e, i) => (
                    <div key={i} className="spread" style={{ borderBottom: "1px solid var(--rule)", paddingBottom: 5 }}>
                      <span className="mono" style={{ fontSize: 10 }}>{e.label} → target</span>
                      <span className="mono" style={{ fontSize: 11, color: e.w >= 0 ? "var(--signal)" : "var(--ember)" }}>{e.w > 0 ? "+" : ""}{fmt(e.w, 3)}</span>
                    </div>
                  ))}
                </div>
              </Panel>
              <Prose>
                <p>
                  The wiring is dense by construction: every stored prompt position feeds
                  its attention block, and every attention block feeds its MLP. That makes
                  the edge count a property of the capture, not evidence that most of the
                  network is off-path. The direct attribution arrays are the readout
                  evidence; they are dense, signed projections, and the largest ones are
                  shown in Figure 14.2.
                </p>
                <p>
                  Do not read either view as a mechanism. Direct attribution passes through
                  the final norm, ignores interactions, and cannot distinguish a component
                  that writes the answer from one that suppresses a competitor. Chapter 13
                  is the causal check, and it often disagrees.
                </p>
              </Prose>
            </div>

            <div className="grid4">
              <Stat k="Nodes" v={entry.nodes.length} s="tokens + blocks" />
              <Stat k="Edges" v={entry.edges.length} s="wiring edges" />
              <Stat k="Target" v={tokenLabel(entry.target) || "—"} s={`${fmt(entry.attr.resid[entry.attr.resid.length - 1] ?? 0, 2)} cumulative logit`} />
              <Stat k="Strongest" v={strongest[0]?.label ?? "—"} s={fmt(strongest[0]?.w ?? 0, 2)} />
            </div>

            <Method>
              <p>
                Nodes are token positions and sublayer outputs. Edges are forward wiring:
                token→attention and attention→MLP. Separately, the capture stores direct
                attributions, the sublayer output projected through the final norm and the
                target token's unembedding row; those arrays feed Figure 14.2.
              </p>
              <p>
                Edge weights are wiring or attention weights, not target contributions.
                Direct attribution has a sign; summing a node's incoming wiring edges does
                not reconstruct it, and attention mixes positions non-linearly.
              </p>
            </Method>

            <Footnotes items={[
              `Captured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); ${ids.length} prompts with graphs.`,
              "Attribution is not causation. The graph is a hypothesis generator for the ablation chapter.",
              "Token nodes are positions, not vocabulary items; the same token at two positions has two nodes.",
              "The layout is deterministic but arbitrary; edge crossings have no meaning.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={14} />
    </>
  );
}
