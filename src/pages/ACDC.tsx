import React from "react";
import { ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat } from "../components/ui";
import { fmt, useOne } from "../lib/atlas";
import type { AcdcAudit } from "../lib/atlas";

const STEPS = [
  {
    n: "01",
    h: "Fix the task and the metric",
    d: "Choose a contrast with a scalar readout — for example the logit difference between the correct and incorrect continuation. The metric is part of the claim: a circuit for “Paris” is not automatically a circuit for factual recall.",
  },
  {
    n: "02",
    h: "Build the candidate graph",
    d: "Nodes are the components you are willing to name: attention heads, MLP blocks, routed experts, shared experts. Edges are residual reads and writes. Restricting to adjacent layers keeps the search finite and states an assumption about composition.",
  },
  {
    n: "03",
    h: "Define the ablation",
    d: "ACDC replaces an edge's activation with a reference value, usually the mean over a task distribution, rather than zero. Mean ablation keeps the scale and sign of the residual stream plausible; it also makes the result depend on the reference distribution.",
  },
  {
    n: "04",
    h: "Score every remaining edge",
    d: "Ablate one edge at a time, rerun the task, and record the change in the metric. This is the expensive step: one forward pass per edge per prompt, before any repeats or batches.",
  },
  {
    n: "05",
    h: "Prune the least important edge",
    d: "Remove the edge with the smallest effect, then re-score what remains. Edges interact, so a single ranking is not a circuit; the loop is what turns attribution into a sparse subgraph.",
  },
  {
    n: "06",
    h: "Validate on held-out prompts",
    d: "Report metric retention, edge count, and a comparison against a random subgraph of the same size. A circuit that only works on the prompts used to find it is a description of those prompts.",
  },
];

const CANDIDATES = [
  {
    k: "Head-level pilot",
    v: "≈7.5k edges",
    d: "27 layers × 16 heads + 1 MoE block = 17 components per layer. Adjacent-layer all-to-all: 17² × 26 ≈ 7,500 edges. One scoring sweep over 27 prompts is ≈2 × 10⁵ forward passes.",
  },
  {
    k: "Expert-level",
    v: "≈1.7 × 10⁵ edges",
    d: "Treating 64 routed + 2 shared experts as separate nodes gives 82 components per layer: 82² × 26 ≈ 1.75 × 10⁵ edges, before the shared expert is handled separately.",
  },
  {
    k: "One scoring sweep",
    v: "not the loop",
    d: "The numbers above are a single pass over the candidate set. ACDC repeats scoring after each prune, so the worst case is quadratic in edges. This is an arithmetic estimate, not a measured runtime.",
  },
];

function Net({ pruned }: { pruned: boolean }) {
  const xs = [42, 128, 214, 278];
  const ys = [42, 92, 142];
  const nodes: { id: string; x: number; y: number }[] = [];
  for (let c = 0; c < xs.length; c++) {
    for (let r = 0; r < ys.length; r++) nodes.push({ id: `${c}-${r}`, x: xs[c], y: ys[r] });
  }
  const kept = new Set([
    "0-1|1-1", "1-1|2-1", "2-1|3-1",
    "0-1|1-0", "1-0|2-1", "0-2|1-2", "1-2|2-2", "2-2|3-2",
  ]);
  const edges: { s: string; t: string }[] = [];
  for (let c = 0; c < xs.length - 1; c++) {
    for (let r = 0; r < ys.length; r++) {
      for (let q = 0; q < ys.length; q++) edges.push({ s: `${c}-${r}`, t: `${c + 1}-${q}` });
    }
  }
  const pos = Object.fromEntries(nodes.map((n) => [n.id, n]));
  return (
    <svg viewBox="0 0 320 184" role="img" aria-label={pruned ? "Illustrative pruned circuit" : "Illustrative candidate graph"}
      style={{ width: "100%", height: "auto", display: "block" }}>
      {edges.map((e) => {
        const on = kept.has(`${e.s}|${e.t}`);
        if (pruned && !on) return null;
        return (
          <line key={`${e.s}|${e.t}`} x1={pos[e.s].x} y1={pos[e.s].y} x2={pos[e.t].x} y2={pos[e.t].y}
            stroke={on ? "var(--signal)" : "var(--rule-2)"} strokeWidth={on ? 1.3 : 0.6}
            opacity={on ? 0.8 : 0.45} />
        );
      })}
      {nodes.map((n) => (
        <circle key={n.id} cx={n.x} cy={n.y} r={5.5}
          fill={pruned && ![...kept].some((k) => k.startsWith(n.id + "|") || k.endsWith("|" + n.id)) ? "var(--paper-3)" : "var(--paper-2)"}
          stroke={n.id === "1-1" || n.id === "2-1" ? "var(--ember)" : "var(--ink-3)"} strokeWidth={1.2} />
      ))}
      {[0, 1, 2, 3].map((c) => (
        <text key={c} x={xs[c]} y={172} textAnchor="middle" className="ticklabel">L{c === 0 ? "in" : c - 1}</text>
      ))}
    </svg>
  );
}

export default function ACDC() {
  const pilotQ = useOne<AcdcAudit>("acdc");
  const pilot = pilotQ.data;

  return (
    <>
      <ChapterHead n="19" part="III" kicker="Automated circuit discovery"
        title={<>Finding the circuit, <em style={{ whiteSpace: "nowrap" }}>edge by edge</em></>}
        standfirst={<>ACDC turns circuit hunting into a search: repeatedly ablate one candidate edge, measure how much the task breaks, remove the least important edge, and stop at a budget. The result is a sparse subgraph — and a set of assumptions that are as important as the graph.</>}
        chips={<><Chip tone="blue">method + pilot</Chip><Chip tone="teal">offline readout</Chip><Chip tone="ember">causal follow-up: ch 23</Chip></>}
      />

      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="Search object" v="Edges" s="not just components" />
        <Stat k="Ablation" v="Mean reference" s="distribution-dependent" />
        <Stat k="Output" v="Sparse subgraph" s="a hypothesis, then test it" />
      </div>

      <Prose>
        <p className="lede">
          Attribution ranks components by their direct write. A circuit needs a stronger
          object: a <em>path</em> through the computation whose removal breaks the behaviour
          and whose retention preserves it. ACDC (Conmy et al., 2023) automates the search for
          that object by treating every candidate edge as an intervention.
        </p>
        <p>
          The algorithm is deliberately simple. Start with a graph of components and edges,
          define a task metric, and ablate edges one at a time with a mean activation. The
          edge whose removal hurts least is pruned. Repeat until the graph has the desired
          number of edges. The paper's contribution is not a new attribution score; it is the
          loop, the edge-level intervention, and the demonstration that a sparse circuit can
          be recovered without hand-picking it.
        </p>
      </Prose>

      <Prose>
        <p className="aside">
          <strong>The loop is a greedy search under a chosen metric.</strong> ACDC does not
          discover a unique circuit; it returns a subgraph that is minimally damaging under a
          particular ablation distribution. Two choices do most of the work: the reference
          used for ablation (mean over which prompts?) and the metric used to score retention
          (logit difference, probability, loss?). A circuit is therefore always indexed to a
          task, a reference distribution and a budget. The useful theoretical discipline is to
          state those three before the search and to compare the result against a random
          subgraph of the same size.
        </p>
      </Prose>

      <Panel title={<><b>Figure 19.1</b> · the ACDC loop</>}
        note="The order matters. Define the metric before looking at the result, and define the ablation before ranking edges. Changing either after seeing the circuit is how a search becomes a story.">
        <div className="stack" style={{ gap: 1, background: "var(--rule)" }}>
          {STEPS.map((s) => (
            <div key={s.n} style={{ background: "var(--paper)", padding: "12px 14px", display: "grid", gridTemplateColumns: "34px 1fr", gap: 12 }}>
              <span className="mono" style={{ fontSize: 10, color: "var(--ember)", paddingTop: 2 }}>{s.n}</span>
              <div>
                <div className="serif" style={{ fontSize: 16.5 }}>{s.h}</div>
                <div style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.6, marginTop: 3 }}>{s.d}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title={<><b>Figure 19.2</b> · the procedure in pseudocode</>}
        note="This is the conceptual loop, not a runnable implementation. A real run needs batched edge ablations, a reference distribution, a held-out split and a seed policy; the naive loop is quadratic in the number of candidate edges.">
        <pre className="code">{`# C: candidate edges, M: task metric, A: mean-ablation operator
# higher M is better; budget is the maximum number of edges to keep
while len(C) > budget:
    scores = {}
    for edge in C:
        ablated = A(C, edge)          # replace edge's write with its mean
        scores[edge] = M(ablated)      # task metric after ablation
    worst = argmin(scores)            # removing it hurts least
    C.remove(worst)

return C`}</pre>
      </Panel>

      <Panel title={<><b>Figure 19.3</b> · candidate graph and a recovered subgraph</>}
        note="Illustrative only — this is a hand-drawn schematic, not an ACDC result on V2-Lite. Left: every adjacent-layer edge is a candidate. Right: a sparse subgraph of the kind ACDC returns; the retained edges still have to be validated on held-out prompts.">
        <div className="grid2" style={{ gap: 20 }}>
          <div>
            <Net pruned={false} />
            <div className="caption" style={{ textAlign: "center" }}>candidate edges</div>
          </div>
          <div>
            <Net pruned />
            <div className="caption" style={{ textAlign: "center" }}>pruned subgraph · schematic</div>
          </div>
        </div>
      </Panel>

      <h2 className="sec">Why the graph is not the claim</h2>
      <Prose>
        <p>
          ACDC returns a subgraph that preserves a metric under a particular ablation. That is
          evidence of <em>necessity under intervention</em>, not a unique decomposition. Mean
          ablation changes the activation distribution; a different reference (zero, shuffled,
          another prompt) can produce a different circuit. The task metric is also a choice: a
          circuit for the correct-token logit difference may not be the circuit for the model's
          actual next-token behaviour under sampling. And edges interact, so pruning one can
          change the importance of another; the loop is an approximation, not an exhaustive
          search.
        </p>
        <p>
          These are not reasons to skip the method. They are reasons to report it precisely:
          task, metric, reference distribution, candidate graph, edge budget, held-out
          prompts, random-subgraph baseline and seed variance. A circuit claim without those
          is a picture of a graph.
        </p>
      </Prose>

      <h2 className="sec">What it would take on V2-Lite</h2>
      <Prose>
        <p>
          The candidate graph is the first constraint. At head level, V2-Lite has 16 attention
          heads and one MoE block per layer, so an adjacent-layer graph has 17 components per
          layer and roughly 7,500 edges across 26 transitions. Scoring every edge once on 27
          prompts is already about 200,000 forward passes. At expert level — 64 routed plus two
          shared experts as separate nodes — the candidate set grows to roughly 175,000 edges.
          Those are arithmetic counts, not measured throughput.
        </p>
        <p>
          The current capture is a single 16 GB T4 at NF4 with 27 prompts. It was sized to
          produce the lens, router, attention, ablation and geometry pages in one pass. ACDC
          needs repeated interventions, a reference distribution and a held-out split; on this
          budget it is a separate experiment, not another hook in the same kernel. Running a
          quantised ACDC would also answer a different question from a bf16 run, because the
          mean activations and the near-ties both change under NF4.
        </p>
      </Prose>

      <Panel title={<><b>Figure 19.4</b> · the candidate budget</>}
        note="The counts assume adjacent-layer all-to-all connectivity and one score per edge per prompt. They exclude embedding/token edges, repeated pruning sweeps and validation runs, so they are lower bounds on the work.">
        <div className="stack" style={{ gap: 1, background: "var(--rule)" }}>
          {CANDIDATES.map((c) => (
            <div key={c.k} style={{ background: "var(--paper)", padding: "12px 14px", display: "grid", gridTemplateColumns: "130px 130px 1fr", gap: 12 }}>
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>{c.k}</span>
              <span className="serif" style={{ fontSize: 17 }}>{c.v}</span>
              <span style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.55 }}>{c.d}</span>
            </div>
          ))}
        </div>
      </Panel>

      <h2 className="sec">What the offline pilot actually does</h2>
      <Prose>
        <p>
          The atlas can still run the <em>shape</em> of the loop on the four captured wiring
          graphs without a GPU. Treat each stored direct attribution as a fixed readout weight,
          propagate it through the captured edges, and greedily remove the edge that changes
          the resulting readout mass least. This tests the graph arithmetic, the pruning order
          and the stability of a candidate subgraph. It is not a model edge ablation: no
          forward pass is rerun, so the result cannot be called a causal ACDC circuit.
        </p>
      </Prose>

      <DataState loading={pilotQ.loading} error={pilotQ.error} what="the offline ACDC pilot">
        {pilot && (
          <>
            <div className="grid3" style={{ marginBottom: 14 }}>
              <Stat k="Readout retained" v={`${fmt(pilot.aggregate.meanRetention * 100, 1)}%`} s={`mean across ${pilot.prompts.length} captured graph prompts`} />
              <Stat k="Edge budget" v={`${pilot.budget} / ${pilot.candidateEdges.join("–")}`} s="kept / candidate edges" />
              <Stat k="Stable edges" v={String(pilot.aggregate.stableEdges)} s="retained in every prompt" />
            </div>
            <Panel title={<><b>Figure 19.5</b> · offline readout pilot, not a causal circuit</>}
              tight
              note={pilot.caveat}>
              <div className="scroll">
                <table className="tbl">
                  <thead>
                    <tr><th>graph prompt</th><th>candidates</th><th>kept</th><th>chain / token</th><th>readout retained</th></tr>
                  </thead>
                  <tbody>
                    {pilot.prompts.map((p) => (
                      <tr key={p.id}>
                        <td className="k">{p.id}</td>
                        <td>{p.candidates}</td>
                        <td>{p.kept}</td>
                        <td>{p.chainKept} / {p.attnKept}</td>
                        <td>{fmt(p.retention * 100, 1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
            <Prose>
              <p className="aside">
                The retention values are close to one another but the retained sets are not a
                sparse circuit: at this budget the loop keeps a large share of the graph and
                the stable set is dominated by the residual chain. That is the negative result
                worth reporting. The pilot shows that the stored graph can be searched, not
                that the search recovers a mechanism. A real run has to replace the readout
                proxy with mean-ablation forward passes and validate the result on held-out
                prompts.
              </p>
            </Prose>
          </>
        )}
      </DataState>

      <h2 className="sec">What would count as a result</h2>
      <Prose>
        <p>
          A credible ACDC page in this atlas would report a circuit at a stated edge budget,
          its task metric on prompts that were not used to find it, and the same metric for a
          random subgraph of equal size. It would show the circuit's stability across seeds and
          across two ablation references. It would name the components in the model's own
          vocabulary — head 7.3, routed expert 41 at layer 12, the shared expert at layer 9 —
          and it would say which of those names survive a change of corpus.
        </p>
        <p>
          None of that is established by the offline pilot above. The causal follow-up in
          chapter 23 runs a real residual-write edge search on V2-Lite, with a discovery
          and a held-out validation prompt. It is still narrower than token-level path
          patching, but it is no longer a graph proxy. The graph on the Circuit page is an
          attribution hypothesis; the ablation on the Ablation page is a node-level
          necessity test; chapter 23 is the edge-level intervention that connects them.
        </p>
      </Prose>

      <Method label="How this is specified">
        <p>
          The procedure follows Conmy et al., <i>Towards Automated Circuit Discovery for
          Mechanistic Interpretability</i> (NeurIPS 2023, arXiv:2304.14997). The candidate counts
          use the official V2-Lite config: 27 layers, 16 attention heads, 64 routed and two
          shared experts. The arithmetic is shown on the page. The only executed ACDC-shaped
          procedure on this page is the offline readout pilot, computed by
          <code> experiments/mechanistic.py</code> from <code>public/data/graph.json</code>.
          The causal residual-write follow-up is chapter 23; this page does not claim its
          result.
        </p>
        <p>
          The mean-ablation detail matters. Zero ablation asks “what if this edge were silent?”;
          mean ablation asks “what if this edge carried the corpus average?”. ACDC uses the
          latter because it keeps the residual stream in distribution, but the choice is a
          modelling assumption, not a neutral preprocessing step.
        </p>
      </Method>

      <Footnotes items={[
        "Conmy et al., <i>Towards Automated Circuit Discovery for Mechanistic Interpretability</i>, arXiv:2304.14997.",
        "The paper's case studies include indirect object identification in GPT-2 small; the method is model-agnostic, but the circuit is not. A V2-Lite circuit would have to be found and validated on V2-Lite.",
        "Mean ablation is not zero ablation. Replacing an edge with the corpus mean preserves scale but can inject information the model would not otherwise have at that position.",
        "The candidate counts assume adjacent-layer all-to-all edges: 17² × 26 ≈ 7,514 and 82² × 26 ≈ 174,824. They are order-of-magnitude arithmetic, not measurements.",
        "The current capture remains a single T4 / NF4 run. The offline readout pilot on this page is explicitly non-causal; the real residual-write edge search is reported separately in chapter 23.",
      ]} />
      <PageFoot index={19} />
    </>
  );
}
