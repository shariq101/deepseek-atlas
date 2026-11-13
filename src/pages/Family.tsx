import React, { useMemo, useState } from "react";
import {
  Family as FamilyData, FamilyMember, compact, fmt, pct, useMeasure, useOne,
} from "../lib/atlas";
import {
  Caption, ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";

const FIELD_LABEL: Record<string, string> = {
  layers: "blocks", hidden: "residual width", heads: "query heads", kvHeads: "KV heads",
  experts: "routed experts", shared: "shared experts", topk: "experts per token",
  moeInter: "expert width", vocab: "vocabulary", context: "context", attention: "attention",
  precision: "precision", mtp: "MTP heads", hash_layers: "hash layers", qLora: "Q lora rank",
  kvLora: "KV lora rank", indexTopk: "index top-k",
};

function SparsityPlot({ members, selected, onSelect }: {
  members: FamilyMember[]; selected: string; onSelect: (id: string) => void;
}) {
  const { ref, width } = useMeasure<HTMLDivElement>();
  const h = 350;
  const m = { t: 22, r: 96, b: 38, l: 58 };
  const x0 = 10, x1 = 12.35, y0 = 9, y1 = 10.85;
  const sx = (v: number) => m.l + ((Math.log10(v) - x0) / (x1 - x0)) * (width - m.l - m.r);
  const sy = (v: number) => h - m.b - ((Math.log10(v) - y0) / (y1 - y0)) * (h - m.t - m.b);
  const xt = [10, 30, 100, 300, 1000].map((v) => v * 1e9);
  const yt = [1, 3, 10, 30, 100].map((v) => v * 1e9);

  return (
    <div ref={ref} style={{ overflowX: "auto" }}>
      <svg width={width} height={h} className="chart">
        {yt.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} className="gridline" />
            <text x={m.l - 6} y={sy(t) + 3} textAnchor="end" className="ticklabel">{compact(t, 0)}</text>
          </g>
        ))}
        {xt.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={h - m.b} y2={h - m.b + 3} stroke="var(--rule-2)" />
            <text x={sx(t)} y={h - m.b + 14} textAnchor="middle" className="ticklabel">{compact(t, 0)}</text>
          </g>
        ))}
        {/* constant-sparsity guides */}
        {[6, 16, 32].map((ratio) => {
          const a = { x: 10 ** x0, y: 10 ** (x0 - Math.log10(ratio)) };
          const b = { x: 10 ** x1, y: 10 ** (x1 - Math.log10(ratio)) };
          return (
            <g key={ratio} opacity={0.6}>
              <line x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)}
                stroke="var(--rule-2)" strokeDasharray="3 4" />
              <text x={sx(b.x) - 4} y={sy(b.y) - 5} textAnchor="end" className="ticklabel">
                {ratio}× sparse
              </text>
            </g>
          );
        })}
        {members.map((mem) => {
          const on = mem.id === selected;
          return (
            <g key={mem.id} onClick={() => onSelect(mem.id)} style={{ cursor: "pointer" }}>
              <circle cx={sx(mem.published_total)} cy={sy(mem.published_active)}
                r={on ? 7 : 4.6} fill={on ? "var(--signal)" : "var(--paper-2)"}
                stroke={on ? "var(--signal)" : "var(--ink)"} strokeWidth={1.3} />
              <text x={sx(mem.published_total) + 9} y={sy(mem.published_active) + 3}
                style={{ fontFamily: "var(--mono)", fontSize: 9.5, fill: on ? "var(--signal)" : "var(--ink-2)" }}>
                {mem.short}
              </text>
            </g>
          );
        })}
        <text x={m.l} y={12} className="axislabel">active parameters (log) ↑</text>
        <text x={width - m.r} y={h - 4} textAnchor="end" className="axislabel">total parameters (log) →</text>
      </svg>
      <div className="hoverbox">
        Dashed lines are constant total/active ratios. The distance from the diagonal is the
        family's central design variable: how much capacity can be stored without being
        paid for on every token.
      </div>
    </div>
  );
}

export default function Family() {
  const { data, loading, error } = useOne<FamilyData>("family");
  const [selected, setSelected] = useState("v4flash");
  const [a, setA] = useState("v3");
  const [b, setB] = useState("v4flash");
  const fam = data;
  const members = fam?.members ?? [];

  const diff = useMemo(() => {
    const ma = members.find((m) => m.id === a);
    const mb = members.find((m) => m.id === b);
    if (!ma || !mb) return [];
    return Object.keys(FIELD_LABEL).map((k) => {
      const va = (ma as unknown as Record<string, unknown>)[k];
      const vb = (mb as unknown as Record<string, unknown>)[k];
      return { k, label: FIELD_LABEL[k], a: va, b: vb, changed: JSON.stringify(va) !== JSON.stringify(vb) };
    });
  }, [a, b, members]);

  const sel = members.find((m) => m.id === selected);

  return (
    <>
      <ChapterHead
        n="01" part="I" kicker="The family"
        title={<>Nine machines, one <em>lineage</em></>}
        standfirst={
          <>
            DeepSeek has shipped the same architectural sentence nine times. Each
            architectural generation changes a noun: <b>latent attention, routed experts,
            a shared expert, sparse indexing, hash routing, compressed attention,
            cross-layer cache</b>. R1 adds a reasoning post-train without changing the
            shape. Laid end to end, the configs read as a research programme rather than
            a product line.
          </>
        }
        chips={
          <>
            <Chip tone="blue">config arithmetic</Chip>
            <Chip>2024 → 2026</Chip>
            <Chip tone="teal">2.4B → 49B active</Chip>
          </>
        }
      />

      <div className="timeline" style={{ "--gens": fam?.generations.length ?? 6 } as React.CSSProperties}>
        {(fam?.generations ?? []).map((g) => (
          <div className="gen" key={g.id}>
            <span className="yr">{g.label.split("·")[1]?.trim()}</span>
            <span className="nm">{g.id}</span>
            <span className="ch">{g.change}</span>
            <span className="ex">{compact(g.active, 0)} active · {g.experts || "—"} experts</span>
          </div>
        ))}
      </div>

      <DataState loading={loading} error={error} what="the family configs">
        <Panel
          title={<><b>Figure 1.1</b> · total against active parameters</>}
          tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>click a point</span>}
          note="Axes are logarithmic. Published totals and active counts are the labs' own; the underlying block arithmetic is checked against them on the budget page. V4.1 has two active counts because its encoder and decoder are asymmetric; 16B (output) is plotted."
        >
          <SparsityPlot members={members} selected={selected} onSelect={setSelected} />
          {sel && (
            <div className="grid4" style={{ marginTop: 12 }}>
              <Stat k="Selected" v={sel.short} s={sel.attention} />
              <Stat k="Total" v={compact(sel.published_total)} s={`${pct(sel.total_error, 1)} config error`} />
              <Stat k="Active" v={compact(sel.published_active)} s={`${pct(sel.active_error, 1)} config error`} />
              <Stat k="Sparsity" v={`${fmt(sel.sparsity, 1)}×`} s="total ÷ active" />
            </div>
          )}
        </Panel>

        <h2 className="sec">The nine checkpoints</h2>
        <Panel
          title={<><b>Figure 1.2</b> · the family, as configured</>}
          tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>{members.length} checkpoints</span>}
          tight
          note="Rows are ordered by release. 'Sparsity' is total / active and is the number that governs inference economics; 'context' is the native maximum, before any external extension."
        >
          <div className="scroll">
            <table className="tbl family-table">
              <thead>
                <tr>
                  <th>model</th><th>released</th><th>total</th><th>active</th><th>sparsity</th>
                  <th>layers</th><th>width</th><th>experts</th><th>top-k</th><th>context</th>
                  <th>attention</th><th>precision</th>
                </tr>
              </thead>
              <tbody>
                {members.map((m) => (
                  <tr key={m.id} className={m.id === selected ? "hi" : ""} onClick={() => setSelected(m.id)} style={{ cursor: "pointer" }}>
                    <td className="k">{m.name}</td>
                    <td>{m.date}</td>
                    <td>{compact(m.published_total)}</td>
                    <td>{compact(m.published_active)}</td>
                    <td>{fmt(m.sparsity, 1)}×</td>
                    <td>{m.layers}</td>
                    <td>{m.hidden}</td>
                    <td>{m.experts || "—"}</td>
                    <td>{m.topk || "—"}</td>
                    <td>{compact(m.context, 0)}</td>
                    <td style={{ fontFamily: "var(--sans)", fontSize: 10.5 }}>{m.attention}</td>
                    <td>{m.precision}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title={<><b>Figure 1.3</b> · what actually changed between two members</>}
          tools={
            <div className="row" style={{ gap: 8 }}>
              <select className="ctl" value={a} onChange={(e) => setA(e.target.value)} style={{ maxWidth: 180 }}>
                {members.map((m) => <option key={m.id} value={m.id}>{m.short}</option>)}
              </select>
              <span className="mini-label" style={{ alignSelf: "center" }}>vs</span>
              <select className="ctl" value={b} onChange={(e) => setB(e.target.value)} style={{ maxWidth: 180 }}>
                {members.map((m) => <option key={m.id} value={m.id}>{m.short}</option>)}
              </select>
            </div>
          }
          tight
          note="Only fields where the two checkpoints differ are shown in colour. A blue value is the larger of the two; an ember value is the smaller. Architecture is more conserved than the marketing suggests."
        >
          <table className="tbl">
            <thead><tr><th>field</th><th>{members.find((m) => m.id === a)?.short}</th><th>{members.find((m) => m.id === b)?.short}</th></tr></thead>
            <tbody>
              {diff.filter((d) => d.changed).map((d) => (
                <tr key={d.k}>
                  <td className="k">{d.label}</td>
                  <td>{String(d.a ?? "—")}</td>
                  <td>{String(d.b ?? "—")}</td>
                </tr>
              ))}
              {diff.filter((d) => d.changed).length === 0 && (
                <tr><td colSpan={3}>These two checkpoints share the same block config. V3 and V3.1 are the clearest example: a behaviour change with no shape change.</td></tr>
              )}
            </tbody>
          </table>
        </Panel>

        <h2 className="sec">Four design moves</h2>
        <Prose>
          <p>
            <strong>V3 made the expert count the scale axis.</strong> It moved from
            V2's 160 routed experts to 256, used one shared expert, activated 8, and
            replaced the auxiliary router loss with an auxiliary-loss-free balancing
            bias. The result is 671B total against 37B active: an 18× sparsity that
            made a frontier model serveable at dense-13B-like compute.
          </p>
          <p>
            <strong>V3.2 made attention sparse in the index.</strong> A 64-head,
            128-dimensional indexer scores the whole context and picks the top 2,048
            keys; the main 128 heads attend only to those. The KV cache still exists,
            but the quadratic term in compute does not grow with it.
          </p>
          <p>
            <strong>V4 made compression the organising principle.</strong> Its 43
            layers alternate compressed-sparse attention (4:1) with heavily
            compressed attention (128:1), with a 128-token sliding window at the
            edges. Three of its layers route by a hash of the token id rather than a
            learned gate, removing the router from the first three blocks entirely.
            Experts are stored in FP4; everything else is FP8.
          </p>
          <p>
            <strong>V4.1 made the cache a shared object.</strong> Only four source
            layers store KV; the rest read it, and the model is split into a causal
            encoder (8B active) and decoder (16B active) with three MTP heads. The
            headline is not parameter count but the ratio: the lab claims 1/4 the HBM
            and 1/8 the SSD traffic of V4-Flash.
          </p>
        </Prose>

        <div className="grid3">
          <Stat k="V3 sparsity" v="18.3×" s="671B / 36.6B computed" />
          <Stat k="V4-Pro sparsity" v="30.6×" s="1.6T / 51.5B computed" />
          <Stat k="V4.1 sparsity" v="32.5×" s="552B / 17.0B computed" />
        </div>

        <Panel
          title={<><b>Figure 1.4</b> · does the arithmetic reproduce the published totals?</>}
          tight
          note="Relative error between the parameter count computed from named tensor shapes and the lab's published total. There is no fitted constant: the residual comes from details the config does not expose (fused projections, indexer, MTP accounting)."
        >
          <table className="tbl">
            <thead><tr><th>model</th><th>published</th><th>computed</th><th>error</th><th>active (pub.)</th><th>active (comp.)</th><th>error</th></tr></thead>
            <tbody>
              {members.map((m) => (
                <tr key={m.id}>
                  <td className="k">{m.short}</td>
                  <td>{compact(m.published_total, 2)}</td>
                  <td>{compact(m.computed_total, 2)}</td>
                  <td style={{ color: Math.abs(m.total_error) < 0.01 ? "var(--teal)" : "var(--ember)" }}>{pct(m.total_error, 1)}</td>
                  <td>{compact(m.published_active, 2)}</td>
                  <td>{compact(m.computed_active, 2)}</td>
                  <td style={{ color: Math.abs(m.active_error) < 0.02 ? "var(--teal)" : "var(--ember)" }}>{pct(m.active_error, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Method label="What is computed, and from what">
          <p>
            Each member is read from its own <code>config.json</code> (sources below).
            Attention is counted as Q, KV-down, KV-up and O projections; every MoE
            layer as <code>n_routed × 3 · d · moe_inter</code> plus the shared expert
            and gate; dense layers as <code>3 · d · intermediate</code>; embeddings
            and an untied head as <code>vocab · d</code> each. The published totals
            exclude MTP blocks, and so does this count, otherwise the comparison
            would not be like for like.
          </p>
          <p>
            V4's grouped low-rank output projection is the one place the config does
            not name every factor. It is counted as{" "}
            <code>heads · v · o_lora + groups · o_lora · d</code> and flagged in the
            architecture chapter. V4.1's 8B encoder and 16B decoder are reported
            separately; only the decoder number is used in the sparsity ratio.
          </p>
        </Method>

        <Footnotes
          items={[
            ...(fam?.sources ?? []).map((s) => `<a href="${s.url}" target="_blank" rel="noreferrer">${s.name}</a>`),
            "“Active” counts every parameter a token touches: all attention and norm weights, the shared expert, the selected routed experts, the gate, and the unembedding. It excludes the embedding lookup, which is a memory read rather than a matmul.",
            "V3 and V3.1 have identical block configs. The diff table will show no changed fields for that pair; that is the intended reading, not a bug.",
          ]}
        />
      </DataState>
      <PageFoot index={1} />
    </>
  );
}
