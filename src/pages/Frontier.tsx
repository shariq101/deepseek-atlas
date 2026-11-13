import React, { useMemo, useState } from "react";
import {
  Frontier as FrontierData, FrontierMember, LayerPlan, bytes, compact, fmt, useMeasure, useOne,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { LineChart, Series } from "../components/charts";

function PlanStrip({ m }: { m: FrontierMember }) {
  const L = m.plan.length;
  const cw = Math.max(5, Math.min(20, 1000 / L));
  const x = (l: number) => 16 + l * cw;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={Math.max(1000, 16 + L * cw)} height={94} className="chart">
        {m.plan.map((p) => {
          const col = p.attn === "hca" ? "#15246e" : p.attn === "csa" ? "#2b46f0" : p.attn === "dsa" ? "#6a3f73" : p.attn === "sliding" ? "#d9d5c8" : "#b58a4a";
          return (
            <g key={p.layer}>
              <rect x={x(p.layer)} y={16} width={cw - 1.3} height={30} fill={col} />
              {p.kv_source && <circle cx={x(p.layer) + (cw - 1.3) / 2} cy={57} r={3.6} fill="none" stroke="var(--teal)" strokeWidth={1.4} />}
              {p.hash && <circle cx={x(p.layer) + (cw - 1.3) / 2} cy={70} r={2.6} fill="var(--ember)" />}
              {p.layer % 5 === 0 && <text x={x(p.layer) + cw / 2} y={88} textAnchor="middle" className="ticklabel">{p.layer}</text>}
            </g>
          );
        })}
        <text x={0} y={12} className="axislabel">attention type · compression · hash · KV source</text>
      </svg>
    </div>
  );
}

export default function Frontier() {
  const { data, loading, error } = useOne<FrontierData>("frontier");
  const [sel, setSel] = useState("v4flash");
  const members = data?.members ?? [];
  const ctxs = data?.contexts ?? [];
  const { ref, width } = useMeasure<HTMLDivElement>();
  const m = members.find((x) => x.id === sel);

  const series = useMemo<Series[]>(() => {
    const picks = ["v3", "v32", "v4flash", "v4pro", "v41flash"];
    const colors: Record<string, string> = {
      v3: "var(--ink-3)", v32: "var(--plum)", v4flash: "var(--ember)",
      v4pro: "var(--signal)", v41flash: "var(--teal)",
    };
    return picks.flatMap((id) => {
      const x = members.find((y) => y.id === id);
      if (!x) return [];
      return [{
        color: colors[id], label: x.short, width: id === "v41flash" ? 2.2 : 1.7,
        values: ctxs.map((c) => Math.log10(x.flops[String(c)] ?? 1)),
      }];
    });
  }, [members, ctxs]);

  const effectiveKeys = (x: FrontierMember) =>
    (x.flops["1048576"] ?? 0) / (x.layers * 2 * x.heads * (x.qk + x.v));

  return (
    <>
      <ChapterHead
        n="05" part="I" kicker="Long context"
        title={<>Attention at one <em>million</em> tokens</>}
        standfirst={
          <>
            A 1M-token context is not a feature; it is three budgets at once —{" "}
            <b>compute</b>, <b>cache</b> and <b>retrieval precision</b>. DeepSeek's last
            three generations each attack a different one: sparse indexing, sequence
            compression, and shared memory.
          </>
        }
        chips={<><Chip tone="blue">config arithmetic</Chip><Chip>4k → 1M</Chip><Chip tone="teal">DSA · CSA · HCA</Chip></>}
      />

      <DataState loading={loading} error={error} what="the long-context arithmetic">
        <Panel
          title={<><b>Figure 5.1</b> · the cost of attending, by context length</>}
          tools={<Legend items={series.map((s) => ({ c: s.color, label: s.label ?? "", line: true }))} />}
          note="Forward attention FLOPs per token, logarithmic both axes. Full MLA (V3) is the straight diagonal; DSA bends it by reading only 2,048 keys; the V4 hybrid bends it further by compressing keys. A flat line here would mean attention cost is independent of context, which no exact method achieves."
        >
          <div ref={ref} style={{ overflowX: "auto" }}>
            <LineChart
              width={Math.min(Math.max(width, 380), 860)} height={360}
              series={series}
              yDomain={[9.5, 16.6]}
              xTicks={[0, 1, 2, 3, 4]}
              xFormat={(i) => compact(ctxs[Math.round(i)] ?? 0, 0)}
              yTicks={[10, 12, 14, 16]}
              yFormat={(v) => `${compact(10 ** v, 0)}`}
              xLabel="context length" yLabel="attention FLOPs / token (log)"
            />
          </div>
          <div className="caption">
            At 1M tokens, a full-attention V3 block performs roughly{" "}
            <b>{compact((members.find((x) => x.id === "v3")?.flops["1048576"] ?? 0), 0)}</b>{" "}
            FLOPs of attention per token. The same width under DSA is{" "}
            <b>{compact((members.find((x) => x.id === "v32")?.flops["1048576"] ?? 0), 0)}</b>.
            The difference is the indexer's 2,048-key choice.
          </div>
        </Panel>

        <h2 className="sec">Three ways to avoid the square</h2>
        <div className="grid3">
          <div style={{ borderTop: "1px solid var(--ink)", paddingTop: 9 }}>
            <div className="mini-label">V3.2 · sparse index</div>
            <div style={{ fontFamily: "var(--serif)", fontSize: 24, margin: "4px 0 6px" }}>DSA</div>
            <p style={{ fontSize: 12.5, color: "var(--ink-2)", lineHeight: 1.6 }}>
              A 64-head indexer scores every key at 128 dimensions, then the main
              heads attend only the top {members.find((x) => x.id === "v32")?.indexTopk ?? 2048}.
              The indexer is still linear in context; the expensive heads are not.
            </p>
            <div className="mono" style={{ fontSize: 9, color: "var(--ember)" }}>
              cost ≈ L · (2·H_idx·d_idx·T + 2·H·(qk+v)·k)
            </div>
          </div>
          <div style={{ borderTop: "1px solid var(--ink)", paddingTop: 9 }}>
            <div className="mini-label">V4 · compression</div>
            <div style={{ fontFamily: "var(--serif)", fontSize: 24, margin: "4px 0 6px" }}>CSA / HCA</div>
            <p style={{ fontSize: 12.5, color: "var(--ink-2)", lineHeight: 1.6 }}>
              Keys are pooled along the sequence: 4 tokens per key in CSA, 128 in HCA.
              Layers alternate the two, and a 128-token sliding window preserves local
              detail. Compression is learned, not a fixed stride.
            </p>
            <div className="mono" style={{ fontSize: 9, color: "var(--ember)" }}>
              cost ≈ Σ_l 2·H·(qk+v)·T / r_l
            </div>
          </div>
          <div style={{ borderTop: "1px solid var(--ink)", paddingTop: 9 }}>
            <div className="mini-label">V4.1 · shared memory</div>
            <div style={{ fontFamily: "var(--serif)", fontSize: 24, margin: "4px 0 6px" }}>CED</div>
            <p style={{ fontSize: 12.5, color: "var(--ink-2)", lineHeight: 1.6 }}>
              A causal encoder (8B active) reads the input; a decoder (16B active)
              writes the output. Only four source layers store KV, and the rest read
              that cache. The saving is in HBM traffic, not in FLOPs alone.
            </p>
            <div className="mono" style={{ fontSize: 9, color: "var(--ember)" }}>
              cache ≈ Σ over 4 source layers (d_head + rope)·T·bytes
            </div>
          </div>
        </div>

        <div className="split" style={{ marginTop: 22 }}>
          <Panel
            title={<><b>Figure 5.2</b> · the depth plan of the hybrid stack</>}
            tools={
              <select className="ctl" value={sel} onChange={(e) => setSel(e.target.value)} style={{ maxWidth: 170 }}>
                {members.filter((x) => x.compress).map((x) => <option key={x.id} value={x.id}>{x.short}</option>)}
              </select>
            }
            note="Top band: attention type. A teal ring is a layer that stores KV for others; an ember dot is a hash-routed MoE layer. V4.1's 4 source layers are visible as four rings among forty blocks."
          >
            {m ? <PlanStrip m={m} /> : null}
            {m && (
              <div className="grid3" style={{ marginTop: 12 }}>
                <Stat k="Attention" v={m.attention.split(" ")[0]} s={m.attention} />
                <Stat k="At 1M / token" v={`${compact(m.flops["1048576"] / 1e12, 2)}T`} s="attention FLOPs" />
                <Stat k="Effective keys" v={compact(effectiveKeys(m), 0)} s="cost ÷ per-key cost" />
              </div>
            )}
          </Panel>

          <Panel
            title={<><b>Figure 5.3</b> · what each context actually stores</>}
            note="At 1M tokens the cache dominates any single GPU. Cache/token and at-1M are per sequence; the last column is attention TFLOPs per token at 1M, computed from the same layers and heads. The cache ratio against MHA is on the KV-cache page."
          >
            <table className="tbl">
              <thead><tr><th>model</th><th>cache / token</th><th>at 1M</th><th>attn TFLOP</th></tr></thead>
              <tbody>
                {members.map((x) => (
                  <tr key={x.id} className={x.id === sel ? "hi" : ""} onClick={() => setSel(x.id)} style={{ cursor: "pointer" }}>
                    <td className="k">{x.short}</td>
                    <td>{bytes(x.kvPerToken)}</td>
                    <td>{bytes(x.kv1m)}</td>
                    <td>{fmt((x.flops["1048576"] ?? 0) / 1e12, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>

        <Prose>
          <p>
            The important distinction is between <strong>reading less</strong> and{" "}
            <strong>storing less</strong>. DSA reads less: its index picks a fixed
            number of keys, so the main attention cost stops growing with context. CSA
            and HCA store less at a coarser resolution: they pool keys, so every token
            still attends globally, but to a blurred summary. Shared KV stores one copy
            instead of many: it saves bandwidth without changing what any single layer
            can see.
          </p>
          <p>
            None of these is free. Top-k selection can miss the one key that matters; a
            128:1 summary cannot preserve a verbatim quotation; a shared cache forces
            layers to agree on what is worth storing. The measured retrieval chapter
            (17) is deliberately on V2-Lite, which has none of these mechanisms, because
            it establishes the baseline those mechanisms are trying to preserve.
          </p>
        </Prose>

        <Method label="How the long-context numbers are computed">
          <p>
            For every member, attention FLOPs per token are evaluated at five context
            lengths. Full MLA uses <code>L · 2 · H · (qk + v) · T</code>. DSA adds the
            indexer term <code>2 · H_idx · d_idx · T</code> and replaces the main term's
            <code>T</code> with <code>min(T, 2048)</code>. V4 sums each layer's
            compression ratio, with ratio-zero layers bounded by the 128-token sliding
            window. These are forward-pass arithmetic counts, not wall-clock timings.
          </p>
          <p>
            V4.1's 8B/16B split is not modelled layer by layer; the figure uses the
            decoder's 40-layer text stack. Its four KV source layers are counted for
            storage, and the other layers are assumed to read the shared cache at no
            additional storage cost.
          </p>
        </Method>

        <Footnotes
          items={[
            "FLOPs are not latency. A sparse attention kernel can be slower than a dense one at short context because of index and gather overhead; the crossover is hardware-specific and is not measured here.",
            "The effective-key figure is a derived quantity: total attention FLOPs divided by the per-key cost of one head at one layer. It is a convenient summary, not an architectural parameter.",
            "V4's compression ratios are learned; the arithmetic treats them as fixed pooling factors. The real kernel may fuse compression with attention, changing the constant but not the scaling.",
            "Context length is the native maximum in the config. Actual serving limits depend on KV budget, batch size and the provider's policy.",
          ]}
        />
      </DataState>
      <PageFoot index={5} />
    </>
  );
}
