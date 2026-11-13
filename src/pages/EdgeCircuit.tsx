import React, { useMemo } from "react";
import { CausalEdge, EdgeCircuit as EdgeData, fmt, pct, useOne } from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";

function CircuitDiagram({ data }: { data: EdgeData }) {
  const L = Math.max(...data.keptEdges.map((e) => e.l), ...data.removedEdges.map((e) => e.l), 0) + 1;
  const W = 760;
  const H = 300;
  const padX = 28;
  const x = (l: number) => padX + (l / Math.max(L - 1, 1)) * (W - padX * 2);
  const ya = 80;
  const ym = 162;
  const kept = new Set(data.keptEdges.map((e) => e.id));
  const edgeMap = new Map<string, CausalEdge>();
  for (const e of [...data.keptEdges, ...data.removedEdges]) edgeMap.set(e.id, e);
  const edges = [...edgeMap.values()];
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={W} height={H} className="chart" role="img" aria-label="Causal residual-write circuit before and after pruning">
        <defs>
          <marker id="edge-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--ink-3)" />
          </marker>
        </defs>
        {Array.from({ length: L }, (_, l) => (
          <g key={`nodes-${l}`}>
            <line x1={x(l)} x2={x(l)} y1={28} y2={H - 38} stroke="var(--rule)" strokeDasharray="2 4" />
            <circle cx={x(l)} cy={ya} r={5} fill={kept.has(`a${l}->m${l}`) ? "var(--signal)" : "var(--paper-2)"} stroke={kept.has(`a${l}->m${l}`) ? "var(--signal)" : "var(--ember)"} />
            <circle cx={x(l)} cy={ym} r={5} fill={kept.has(`m${l}->a${l + 1}`) || kept.has(`m${l}->readout`) ? "var(--signal)" : "var(--paper-2)"} stroke="var(--ink-3)" />
            {(l % 5 === 0 || l === L - 1) && <text x={x(l)} y={H - 20} textAnchor="middle" className="ticklabel">L{l}</text>}
          </g>
        ))}
        <text x={padX} y={20} className="mini-label">RESIDUAL-WRITE GRAPH · kept blue / zero-ablated ember</text>
        <text x={padX - 12} y={ya + 3} textAnchor="end" className="ticklabel">a</text>
        <text x={padX - 12} y={ym + 3} textAnchor="end" className="ticklabel">m</text>
        {edges.map((e) => {
          const on = kept.has(e.id);
          if (e.id.startsWith("a")) {
            return <line key={e.id} x1={x(e.l)} y1={ya + 5} x2={x(e.l)} y2={ym - 6}
              stroke={on ? "var(--signal)" : "var(--ember)"} strokeWidth={on ? 2 : 1} opacity={on ? 0.9 : 0.35} />;
          }
          if (e.t === "readout") {
            return <line key={e.id} x1={x(e.l)} y1={ym + 5} x2={x(e.l)} y2={230}
              stroke={on ? "var(--signal)" : "var(--ember)"} strokeWidth={on ? 2 : 1} opacity={on ? 0.9 : 0.35} markerEnd="url(#edge-arrow)" />;
          }
          const target = Number(e.t.replace("a", ""));
          return <line key={e.id} x1={x(e.l)} y1={ym + 5} x2={x(target)} y2={ya - 6}
            stroke={on ? "var(--signal)" : "var(--ember)"} strokeWidth={on ? 2 : 1} opacity={on ? 0.9 : 0.35} />;
        })}
        <line x1={padX} y1={250} x2={W - padX} y2={250} stroke="var(--rule-2)" />
        <text x={padX} y={270} className="blocksub">attention write → MLP read</text>
        <text x={padX + 220} y={270} className="blocksub">MLP write → next attention</text>
        <text x={W - padX} y={270} textAnchor="end" className="blocksub">final MLP → readout</text>
      </svg>
    </div>
  );
}

export default function EdgeCircuitPage() {
  const q = useOne<EdgeData>("edges");
  const data = q.data;
  const kept = data?.keptEdges.length ?? 0;
  const removed = data?.removedEdges.length ?? 0;
  const steps = useMemo(() => [...(data?.steps ?? [])].reverse(), [data]);

  return (
    <>
      <ChapterHead
        n="23" part="IV" kicker="Causal edge search"
        title={<>A circuit, <em>ablated for real</em></>}
        standfirst={
          <>
            This is the edge-ablation run the earlier ACDC chapter could only specify.
            Start with a 54-edge residual-write graph, remove the edge whose removal hurts
            the target least, rerun the model, and repeat. Then ablate the entire removed
            set on a prompt that was not used to find it.
          </>
        }
        chips={
          <>
            <Chip tone="blue">real edge ablation</Chip>
            <Chip>V2-Lite · NF4</Chip>
            <Chip tone="ember">zero reference</Chip>
          </>
        }
      />

      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="Candidate graph" v={data ? `${data.budget} / ${data.candidateCount}` : "—"} s="kept / candidate edges" />
        <Stat k="Discovery retention" v={data ? pct(data.discoveryRetention) : "—"} s={data?.prompts.discovery ?? "awaiting run"} />
        <Stat k="Held-out retention" v={data?.validation ? pct(data.validation.retention) : "—"} s={data?.prompts.validation ?? "awaiting validation"} />
      </div>

      <Prose>
        <p>
          The proxy in chapter 19 propagated stored direct attributions through a graph.
          This page does the thing the proxy could not: every candidate edge is a module
          whose output is zeroed for one real forward pass. The score is the target-token
          probability at the final position. The loop removes the least damaging edge,
          updates the candidate set, and reruns. A circuit is the set that remains.
        </p>
        <p className="aside">
          The graph is deliberately smaller than the model. It contains only residual
          writes: attention → MLP and MLP → next attention, plus the final MLP → readout.
          Token-to-attention edges and path patching are not included. That makes the
          result an edge-level causal statement about component writes, not a complete
          account of every path through a token.
        </p>
      </Prose>

      <DataState loading={q.loading} error={q.error} what="the causal edge capture">
        {data && (
          <>
            <Panel
              title={<><b>Figure 23.1</b> · the graph after the search</>}
              note={`Blue edges survived the ${data.budget}-edge budget; ember edges were zero-ablated. Discovery prompt: ${data.prompts.discovery}. Held-out prompt: ${data.prompts.validation}. The held-out number ablates every removed edge at once, not one at a time.`}
            >
              <CircuitDiagram data={data} />
              <div className="grid3" style={{ marginTop: 12 }}>
                <Stat k="Kept" v={String(kept)} s="edges remaining" />
                <Stat k="Removed" v={String(removed)} s="zero-ablated writes" />
                <Stat k="Discovery Δp" v={fmt(data.discoveryCircuit.prob - data.baseline.prob, 4)} s="circuit minus baseline" />
              </div>
            </Panel>

            <div className="grid2" style={{ alignItems: "start" }}>
              <Panel title={<><b>Figure 23.2</b> · discovery and validation</>} tight
                note="Target probability at the final token. The circuit columns ablate the whole removed set. Discovery is the prompt used to choose the edges; validation is a different prompt.">
                <table className="tbl">
                  <thead><tr><th>prompt</th><th>baseline p</th><th>circuit p</th><th>retention</th><th>margin</th></tr></thead>
                  <tbody>
                    <tr><td className="k">{data.prompts.discovery}</td><td>{fmt(data.baseline.prob, 4)}</td><td>{fmt(data.discoveryCircuit.prob, 4)}</td><td>{pct(data.discoveryRetention)}</td><td>{fmt(data.discoveryCircuit.margin, 3)}</td></tr>
                    {data.validation && (
                      <tr><td className="k">{data.validation.prompt}</td><td>{fmt(data.validation.baseline.prob, 4)}</td><td>{fmt(data.validation.circuit.prob, 4)}</td><td>{pct(data.validation.retention)}</td><td>{fmt(data.validation.circuit.margin, 3)}</td></tr>
                    )}
                  </tbody>
                </table>
                <p className="caption" style={{ marginTop: 10 }}>
                  A retention near 100% would mean the removed edges carried little of the
                  answer. A collapse is evidence that the search found a load-bearing set,
                  but not proof that it is the only one.
                </p>
              </Panel>
              <Panel title={<><b>Figure 23.3</b> · the pruning order</>} tight
                note="Each row is one real intervention. The smallest absolute target-probability change is removed first; the loop then rescores what remains.">
                <div className="scroll" style={{ maxHeight: 360 }}>
                  <table className="tbl">
                    <thead><tr><th>#</th><th>edge</th><th>kind</th><th>Δ target p</th><th>kept</th></tr></thead>
                    <tbody>
                      {steps.map((s) => (
                        <tr key={s.step}>
                          <td>{s.step}</td><td className="k">{s.edge}</td><td>{s.kind}</td>
                          <td>{fmt(s.deltaProb, 4)}</td><td>{s.kept}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </div>

            <Prose>
              <p className="aside">
                The search is greedy, so it is not guaranteed to find the smallest faithful
                circuit. It is also using zero ablation; a mean reference would preserve
                the activation scale and could reorder the edges. The validation prompt is
                a start, not a held-out suite. The result is a real causal experiment with
                a narrow claim: on these two prompts, this set of residual writes is the
                cheapest set to remove under this metric.
              </p>
            </Prose>
          </>
        )}
      </DataState>

      <Method label="How the edge search was run">
        <p>
          The candidate graph has 54 edges: one attention write per layer, one MLP write
          per layer, and one final MLP-to-readout write. At each step every remaining edge
          is zeroed in a separate forward pass; the edge with the smallest absolute change
          in target probability is removed. The loop stops at 24 kept edges. The removed
          set is then applied simultaneously on a different prompt for validation.
        </p>
        <p>
          This is a real model intervention, not a replay of stored attributions. It is not
          mean ablation, not path patching and not a full token-level ACDC. Run with
          <code> STAGES=edges</code> in the capture kernel; the implementation is in
          <code> experiments/suite.py</code>.
        </p>
      </Method>

      <Footnotes items={[
        "Zero ablation removes the component's write entirely. A mean ablation would replace it with a reference activation, which is the ACDC paper's default but is not what this run uses.",
        "The target metric is the probability of the hand-written continuation's first token. A different metric can select a different circuit.",
        "The held-out validation is one prompt, not a corpus. It tests transfer of the removed set, not statistical stability.",
        "The graph is a residual-write chain. Token-to-attention edges, head-level edges and path patching are outside this run.",
      ]} />
      <PageFoot index={23} />
    </>
  );
}
