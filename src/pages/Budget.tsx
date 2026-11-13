import React, { useMemo, useState } from "react";
import {
  Budget as BudgetData, BudgetMember, compact, fmt, pct, useMeasure, useOne,
} from "../lib/atlas";
import {
  Caption, ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { LineChart } from "../components/charts";

function StackRow({ m, max }: { m: BudgetMember; max: number }) {
  const parts = [
    { k: "routed experts", v: m.routedTotal, c: "var(--signal)" },
    { k: "shared expert", v: m.sharedTotal, c: "var(--ember)" },
    { k: "attention", v: m.attn, c: "var(--teal)" },
    { k: "dense FFN", v: m.dense, c: "var(--sand)" },
    { k: "embeddings + head", v: m.embed + m.lmHead, c: "var(--plum)" },
    { k: "norms", v: m.norms, c: "var(--ink-3)" },
  ];
  const total = parts.reduce((a, b) => a + b.v, 0);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "78px 1fr 72px", gap: 10, alignItems: "center", marginBottom: 7 }}>
      <span className="mini-label" style={{ textTransform: "none", letterSpacing: 0 }}>{m.short}</span>
      <div style={{ display: "flex", height: 20, width: `${(total / max) * 100}%`, border: "1px solid var(--rule-2)" }}>
        {parts.map((p) => (
          <span key={p.k} style={{ width: `${(p.v / total) * 100}%`, background: p.c, borderRight: "1px solid var(--paper)" }}
            title={`${p.k}: ${compact(p.v)} (${pct(p.v / total, 1)})`} />
        ))}
      </div>
      <span className="mono" style={{ fontSize: 9, color: "var(--ink-2)", textAlign: "right" }}>{compact(total, 1)}</span>
    </div>
  );
}

export default function Budget() {
  const { data, loading, error } = useOne<BudgetData>("budget");
  const [sel, setSel] = useState("v4pro");
  const members = data?.members ?? [];
  const m = members.find((x) => x.id === sel) ?? members[0];
  const { ref, width } = useMeasure<HTMLDivElement>();

  const activeParts = useMemo(() => {
    if (!m) return [];
    const shared = m.sharedTotal;
    const routed = m.routedTotal * (m.topk / Math.max(1, m.experts));
    return [
      { label: "attention", value: m.attn, color: "var(--teal)" },
      { label: "shared expert", value: shared, color: "var(--ember)" },
      { label: "routed top-k", value: routed, color: "var(--signal)" },
      { label: "dense FFN", value: m.dense, color: "var(--sand)" },
      { label: "unembedding", value: m.lmHead, color: "var(--plum)" },
      { label: "norms", value: m.norms, color: "var(--ink-3)" },
    ].filter((d) => d.value > 0);
  }, [m]);

  const activeTotal = activeParts.reduce((a, b) => a + b.value, 0);

  const flopSeries = useMemo(() => {
    if (!m) return [];
    const ctxs = [1024, 8192, 32768, 131072, 1048576];
    return [
      {
        color: "var(--signal)", label: "this model", width: 2,
        values: ctxs.map((c) => (m.attnFlops[String(c)] ?? 0) / 1e12 + (2 * m.active) / 1e12),
      },
      {
        color: "var(--rule-2)", label: "dense model of same total", dash: "3 3", width: 1.4,
        values: ctxs.map(() => (2 * m.total) / 1e12),
      },
    ];
  }, [m]);

  const ctxLabels = [1024, 8192, 32768, 131072, 1048576];

  return (
    <>
      <ChapterHead
        n="03" part="I" kicker="Parameter budget"
        title={<>Where 1.6 trillion parameters <em>live</em></>}
        standfirst={
          <>
            Sparsity is not one number. A model has a <b>storage</b> cost set by total
            parameters, a <b>compute</b> cost set by active parameters, and a{" "}
            <b>bandwidth</b> cost set by neither. This chapter separates them, for every
            member of the family.
          </>
        }
        chips={<><Chip tone="blue">config arithmetic</Chip><Chip>9 checkpoints</Chip><Chip tone="teal">2.4B → 49B active</Chip></>}
      />

      <DataState loading={loading} error={error} what="the parameter budget">
        <Panel
          title={<><b>Figure 3.1</b> · the parameter budget, decomposed</>}
          note="Each bar is one checkpoint's total parameter count. Routed experts are the pale-blue mass: the capacity that exists but is not applied to most tokens. The rows are scaled to the largest model so that V2-Lite remains visible."
          tools={<Legend items={[
            { c: "var(--signal)", label: "routed" }, { c: "var(--ember)", label: "shared" },
            { c: "var(--teal)", label: "attention" }, { c: "var(--sand)", label: "dense" },
            { c: "var(--plum)", label: "embed" },
          ]} />}
        >
          <div>
            {members.map((x) => <StackRow key={x.id} m={x} max={Math.max(...members.map((y) => y.total))} />)}
          </div>
          <div className="grid3" style={{ marginTop: 16 }}>
            <Stat k="Expert share, V2-Lite" v="94.5%" s="of all parameters" />
            <Stat k="Expert share, V4-Pro" v="98.5%" s="of all parameters" />
            <Stat k="Attention share, V3" v="1.7%" s="of all parameters" />
          </div>
        </Panel>

        <div className="split">
          <Panel
            title={<><b>Figure 3.2</b> · what a token actually pays for</>}
            tools={
              <select className="ctl" value={sel} onChange={(e) => setSel(e.target.value)} style={{ maxWidth: 160 }}>
                {members.map((x) => <option key={x.id} value={x.id}>{x.short}</option>)}
              </select>
            }
            note="Active parameters, decomposed. Attention is paid on every layer; the shared expert is paid on every MoE layer; only the selected experts are paid per token. Across the family, active work is roughly 3–16% of stored parameters."
          >
            <div className="stack" style={{ gap: 7 }}>
              {activeParts.map((p) => (
                <div key={p.label} style={{ display: "grid", gridTemplateColumns: "100px 1fr 72px", gap: 8, alignItems: "center" }}>
                  <span className="mono" style={{ fontSize: 9.5, color: "var(--ink-2)" }}>{p.label}</span>
                  <div style={{ height: 13, background: "rgba(16,18,23,0.04)" }}>
                    <div style={{ height: "100%", width: `${(p.value / activeTotal) * 100}%`, background: p.color }} />
                  </div>
                  <span className="mono" style={{ fontSize: 9, color: "var(--ink-2)", textAlign: "right" }}>{compact(p.value, 1)}</span>
                </div>
              ))}
            </div>
            <div className="grid2" style={{ marginTop: 16 }}>
              <Stat k="Active total" v={compact(activeTotal, 1)} s={m ? `${pct(activeTotal / m.total, 1)} of storage` : "loading"} />
              <Stat k="Routed per token" v={m ? `${m.topk} / ${m.experts}` : "—"} s={`${compact(activeParts.find((p) => p.label === "routed top-k")?.value ?? 0, 1)} active`} />
            </div>
          </Panel>

          <div className="stack">
            <Panel title={<><b>Figure 3.3</b> · the price of a token</>} note="Forward FLOPs per token (2·N) plus attention at the stated context. The dashed line is a dense model with the same storage.">
              <div ref={ref} style={{ overflowX: "auto" }}>
                {m && (
                  <LineChart
                    width={Math.min(Math.max(width, 300), 760)} height={250}
                    series={flopSeries}
                    yDomain={[0, Math.max(...flopSeries.flatMap((s) => s.values)) * 1.1]}
                    xLabel="context (tokens, log-ish spacing)" yLabel="TFLOPs / token"
                    xTicks={[0, 1, 2, 3, 4]} yFormat={(v) => fmt(v, 1)}
                    xFormat={(i) => compact(ctxLabels[Math.round(i)] ?? 0, 0)}
                  />
                )}
              </div>
              <div className="legend" style={{ marginTop: 6 }}>
                <Legend items={[{ c: "var(--signal)", label: "this model", line: true }, { c: "var(--rule-2)", label: "dense equivalent", line: true }]} />
              </div>
            </Panel>
            <div className="grid2">
              <Stat k="Forward FLOPs" v={`${fmt((2 * (m?.active ?? 0)) / 1e12, 1)}T`} s="2 × active, at short context" />
              <Stat k="Dense equivalent" v={`${fmt((2 * (m?.total ?? 0)) / 1e12, 1)}T`} s="2 × total" />
            </div>
          </div>
        </div>

        <h2 className="sec">The exact table</h2>
        <Panel
          title={<><b>Figure 3.4</b> · per-layer and per-expert arithmetic</>}
          tight
          note="All values are products of named shapes. 'Expert size' is the three SwiGLU matrices of one routed expert; 'layer FFN' includes all routed and shared experts."
        >
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr><th>model</th><th>total</th><th>active</th><th>sparsity</th><th>attn / layer</th><th>expert size</th><th>layer FFN</th><th>active share</th></tr>
              </thead>
              <tbody>
                {members.map((x) => (
                  <tr key={x.id} className={x.id === sel ? "hi" : ""} onClick={() => setSel(x.id)} style={{ cursor: "pointer" }}>
                    <td className="k">{x.short}</td>
                    <td>{compact(x.total, 1)}</td>
                    <td>{compact(x.active, 1)}</td>
                    <td>{fmt(x.sparsity, 1)}×</td>
                    <td>{compact(x.attnPerLayer, 2)}</td>
                    <td>{compact(x.perExpert, 2)}</td>
                    <td>{compact(x.routedPerLayer + x.sharedPerLayer, 1)}</td>
                    <td>{pct(x.active / x.total, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Prose>
          <p>
            The family's economics are visible in one ratio. V2-Lite stores 15.7B
            parameters and applies 2.5B; V4-Pro stores 1.6T and applies 51.5B. Storage
            grew 100×, active compute grew 20×. That difference is the entire point of
            sparse experts: capacity scales with data, cost scales with routing.
          </p>
          <p>
            But active parameters are not the whole inference bill. At long context,
            attention FLOPs can dominate; at batch size 1, memory bandwidth can dominate
            both. The next chapter takes the cache apart, because the KV cache — not the
            weights — is what usually stops a long-context request from fitting.
          </p>
        </Prose>

        <Method>
          <p>
            Total parameters are the sum of every matrix and norm gain: Q/KV/O
            projections, every routed expert, the shared expert, the gate, dense
            layers, embeddings, an untied head, and the per-block norms. Active
            parameters replace the full expert set with <code>topk / n_routed</code> of
            it, and keep attention, shared expert and norms.
          </p>
          <p>
            FLOPs use the standard <code>2N</code> forward approximation. The attention
            term is computed separately per architecture: full causal for MLA, a
            2,048-key top-k for DSA, and per-layer compression ratios for CSA/HCA. At
            1M tokens, the attention term is no longer negligible, which is exactly
            what V4's hybrid stack is for.
          </p>
        </Method>

        <Footnotes
          items={[
            "Published active counts differ from the computed ones by 1–9%; the largest gap is V4-Flash, where the config does not expose the indexer's active compute. The published number is used in the prose; the computed one is used in the figures.",
            "The dense equivalent is 2 × total parameters. It is a lower bound on what a dense model of the same capacity would cost, because it ignores the memory traffic of reading all experts.",
            "Expert size counts gate, up and down matrices: 3 × d × moe_intermediate_size. Biases are absent in these checkpoints.",
            "V4.1's 8B encoder / 16B decoder split is reported as a single 17B active figure here; the encoder path applies fewer layers, so its true input-side cost is lower.",
          ]}
        />
      </DataState>
      <PageFoot index={3} />
    </>
  );
}
