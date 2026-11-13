import React from "react";
import { ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat } from "../components/ui";
import { fmt, useOne } from "../lib/atlas";
import type { MechanismAudit } from "../lib/atlas";

const LADDER = [
  {
    n: "01",
    name: "Description",
    asks: "Does this component's activation co-vary with the behaviour?",
    licenses: "A hypothesis, and a reason to intervene.",
    fails: "Correlation, polysemanticity, and the fact that many components are correlated with everything.",
  },
  {
    n: "02",
    name: "Attribution",
    asks: "How much does this component's direct write move the target logit?",
    licenses: "A ranked path, with signed contributions and a baseline.",
    fails: "It is local and first-order. A component can matter through another component, not directly.",
  },
  {
    n: "03",
    name: "Necessity",
    asks: "What happens if I ablate this component and rerun the task?",
    licenses: "A causal claim about this checkpoint, this corpus, this metric.",
    fails: "The model can compensate; mean ablation changes the distribution; necessity is not sufficiency.",
  },
  {
    n: "04",
    name: "Path specificity",
    asks: "Does the effect survive when I patch only this path?",
    licenses: "A claim that the edge, not just the node, carries the behaviour.",
    fails: "Path patching is expensive, and a successful patch can still have many equivalent explanations.",
  },
  {
    n: "05",
    name: "Automated circuit",
    asks: "Can an algorithm recover a sparse subgraph that preserves the task?",
    licenses: "A reproducible circuit hypothesis at edge granularity.",
    fails: "The result depends on the task metric, ablation distribution, candidate graph and pruning schedule.",
  },
];

const OBJECTS = [
  {
    k: "Residual stream",
    v: "A shared, additive bus",
    d: "Every head and MLP reads from it and writes back into it. The stream is the only object all layers share, which is what makes composition possible.",
  },
  {
    k: "Attention head",
    v: "Two bilinear circuits",
    d: "QK chooses where to read; OV chooses what to write. Both are linear maps on the residual stream, but the softmax between them makes the read data-dependent.",
  },
  {
    k: "MLP block",
    v: "A key-value memory",
    d: "The first matrix detects directions; the second writes associated directions back. In an MoE, the router decides which memories run for this token.",
  },
  {
    k: "Router",
    v: "A discrete gate",
    d: "The top-k choice is part of the mechanism, not an implementation detail. A circuit that ignores routing is a dense-model circuit pasted onto a sparse model.",
  },
];

export default function Mechanism() {
  const auditQ = useOne<MechanismAudit>("mechanism");
  const audit = auditQ.data;

  return (
    <>
      <ChapterHead n="18" part="III" kicker="Mechanistic interpretation"
        title={<>What counts as a <em>mechanism?</em></>}
        standfirst={<>A mechanism is not a story about what a model “must” be doing. It is a small, causally tested set of components and paths that explains a behaviour, together with the interventions that would make the claim fail.</>}
        chips={<><Chip tone="blue">method + audit</Chip><Chip>no new forward passes</Chip><Chip>V2-Lite evidence base</Chip></>}
      />

      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="Object of study" v="Components + paths" s="not neurons alone" />
        <Stat k="Evidence" v="Intervention" s="not activation alone" />
        <Stat k="Unit of claim" v="A failing test" s="stated before the result" />
      </div>

      <Prose>
        <p className="lede">
          Mechanistic interpretability asks for an explanation at the granularity of the
          computation: which attention heads, MLP blocks, experts and residual paths carry a
          behaviour, and how their writes compose. The useful contrast is with an explanation
          that merely fits: “the model attends to France because the answer is Paris” can be
          true and still say nothing about the mechanism.
        </p>
        <p>
          The strongest version of the claim has three parts. <strong>Localisation</strong>:
          the behaviour depends on a small set of components. <strong>Causality</strong>:
          removing or patching them changes the behaviour in the predicted direction.{" "}
          <strong>Specificity</strong>: the effect travels through the proposed paths, not
          through a generic change in the residual stream. Each part needs a different
          experiment, and a result that establishes one does not establish the others.
        </p>
      </Prose>

      <Prose>
        <p className="aside">
          <strong>Three burdens, not one.</strong> A circuit is <em>faithful</em> when replacing
          it with an equivalent subgraph preserves the task metric; <em>complete</em> when it
          includes the components the behaviour actually needs; and <em>minimal</em> when it
          contains no component that can be removed without changing the metric. These can pull
          against each other. A large subgraph is easy to make faithful and hard to make
          minimal; a tiny subgraph is easy to make minimal and may fail completeness. Causal
          scrubbing makes the first burden explicit by asking whether a resampled replacement
          preserves the behaviour, not merely whether an activation looks important.
        </p>
      </Prose>

      <Panel title={<><b>Figure 18.1</b> · the evidence ladder</>}
        note="Each rung licenses a different sentence. The ladder is not a maturity scale for a single paper; it is a checklist for the claim being made. Most interpretability results stop at rung two or three.">
        <div className="stack" style={{ gap: 1, background: "var(--rule)" }}>
          {LADDER.map((r) => (
            <div key={r.n} style={{ background: "var(--paper)", padding: "13px 14px", display: "grid", gridTemplateColumns: "34px 1fr", gap: 12 }}>
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)", paddingTop: 2 }}>{r.n}</span>
              <div>
                <div className="spread" style={{ alignItems: "baseline", gap: 10 }}>
                  <span className="serif" style={{ fontSize: 18 }}>{r.name}</span>
                  <span className="mono" style={{ fontSize: 8.5, letterSpacing: "0.07em", textTransform: "uppercase", color: "var(--signal)" }}>{r.licenses}</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.55, marginTop: 4 }}>{r.asks}</div>
                <div className="caption" style={{ marginTop: 3 }}>Fails when: {r.fails}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <h2 className="sec">The objects a DeepSeek mechanism is made of</h2>
      <Prose>
        <p>
          The transformer-circuits framing is useful because it names the objects precisely.
          The residual stream is a communication channel; an attention head is a pair of
          linear maps with a softmax in between; an MLP is a key-value memory. A circuit is a
          subgraph of these objects, not a metaphor about “attention” in general.
        </p>
      </Prose>

      <div className="grid2">
        {OBJECTS.map((o) => (
          <Panel key={o.k} title={<><b>{o.k}</b> · {o.v}</>}>
            <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)", margin: 0 }}>{o.d}</p>
          </Panel>
        ))}
      </div>

      <Panel title={<><b>Figure 18.2</b> · a head reads, then writes</>}
        note="Schematic, not a measurement. The QK circuit sets the attention pattern from the current residual state; the OV circuit writes a weighted sum of value vectors back into the stream. In MLA the keys and values are cached as a compressed latent, but the query/key and output/value computations are still there.">
        <svg viewBox="0 0 640 240" role="img" aria-label="Schematic of an attention head reading from and writing to a residual stream"
          style={{ width: "100%", height: "auto", display: "block" }}>
          <defs>
            <marker id="mech-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--ink-3)" />
            </marker>
          </defs>
          <line x1="24" y1="150" x2="616" y2="150" stroke="var(--ink)" strokeWidth="1.2" markerEnd="url(#mech-arrow)" />
          <text x="24" y="174" className="ticklabel">residual stream r<tspan baselineShift="sub">l</tspan></text>
          {["The", "capital", "of", "France", "is"].map((t, i) => (
            <g key={t}>
              <circle cx={72 + i * 46} cy="150" r="4" fill="var(--paper-2)" stroke="var(--ink-3)" />
              <text x={72 + i * 46} y="136" textAnchor="middle" className="ticklabel">{t}</text>
            </g>
          ))}
          <rect x="214" y="38" width="150" height="58" fill="var(--paper-2)" stroke="var(--rule-2)" />
          <text x="228" y="59" className="mono" style={{ fontSize: 10, fill: "var(--ink)" }}>attention head h</text>
          <text x="228" y="75" className="mono" style={{ fontSize: 8.5, fill: "var(--ink-3)" }}>QK: where to read</text>
          <text x="228" y="88" className="mono" style={{ fontSize: 8.5, fill: "var(--ink-3)" }}>OV: what to write</text>
          <path d="M286,150 C286,120 286,100 286,96" fill="none" stroke="var(--ink-3)" strokeWidth="1.1" markerEnd="url(#mech-arrow)" />
          <path d="M300,96 C300,120 300,130 300,150" fill="none" stroke="var(--signal)" strokeWidth="1.1" markerEnd="url(#mech-arrow)" />
          <text x="306" y="126" className="ticklabel">read</text>
          <text x="306" y="104" style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--signal)" }}>write</text>
          <rect x="428" y="38" width="166" height="58" fill="var(--paper-2)" stroke="var(--rule-2)" />
          <text x="442" y="59" className="mono" style={{ fontSize: 10, fill: "var(--ink)" }}>MLP / MoE</text>
          <text x="442" y="75" className="mono" style={{ fontSize: 8.5, fill: "var(--ink-3)" }}>router → 6 of 64 routed</text>
          <text x="442" y="88" className="mono" style={{ fontSize: 8.5, fill: "var(--ink-3)" }}>+ 1 shared, always on</text>
          <path d="M511,150 C511,120 511,100 511,96" fill="none" stroke="var(--ink-3)" strokeWidth="1.1" markerEnd="url(#mech-arrow)" />
          <path d="M525,96 C525,120 525,130 525,150" fill="none" stroke="var(--ember)" strokeWidth="1.1" markerEnd="url(#mech-arrow)" />
          <text x="530" y="126" className="ticklabel">read</text>
          <text x="530" y="104" style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ember)" }}>write</text>
          <text x="24" y="30" className="ticklabel">component</text>
          <text x="24" y="44" className="ticklabel">reads r_l, writes back</text>
        </svg>
      </Panel>

      <h2 className="sec">From components to a circuit</h2>
      <Prose>
        <p>
          A useful mechanism is <em>sparse</em>, <em>causal</em> and <em>portable</em>. Sparse
          because a circuit that contains half the model explains nothing; causal because the
          components have been removed, patched or ablated; portable because the same circuit
          should recur across prompts that exercise the same computation, not just the one
          example that motivated it.
        </p>
        <p>
          For DeepSeek there are two extra objects in every claim. The <strong>router</strong>{" "}
          is a discrete decision, so a circuit that names “the MLP at layer 8” is incomplete
          unless it says which experts ran and why. The <strong>shared expert</strong> is always
          on, so it is easy to miss: it contributes to every token and may carry the generic
          computation that a routed circuit assumes is background. Both are measurable with the
          same hooks, but they change what a “component” means.
        </p>
      </Prose>

      <h2 className="sec">What the existing capture already shows</h2>
      <Prose>
        <p>
          The atlas already contains two different measurements of the same layers: a
          first-order attribution of each component's direct write to the target logit, and a
          causal mean-ablation delta. They are not the same question. Attribution asks where a
          component writes; ablation asks whether the task still works without it. Comparing
          their layer rankings is a cheap audit of whether the atlas's “important layer” story
          survives a necessity check.
        </p>
      </Prose>

      <DataState loading={auditQ.loading} error={auditQ.error} what="the offline mechanism audit">
        {audit && (
          <>
            <div className="grid3" style={{ marginBottom: 14 }}>
              <Stat k="Attention ranking" v={fmt(audit.agreement.attnSpearman, 2)} s="Spearman ρ · direct attribution vs ablation" />
              <Stat k="MLP ranking" v={fmt(audit.agreement.mlpSpearman, 2)} s="Spearman ρ · 27 layers" />
              <Stat k="Top-five overlap" v={`${audit.agreement.attnTop5Overlap}/5 · ${audit.agreement.mlpTop5Overlap}/5`} s="attention · MLP" />
            </div>
            <Panel title={<><b>Figure 18.3</b> · attribution is not necessity</>}
              tight
              note={`Offline audit only: ${audit.nPrompts} prompts for direct attribution, ${audit.nAblationPrompts} for ablation, both from the same V2-Lite capture. The prompt subsets differ, so this is a descriptive rank comparison, not a significance test. No new forward passes were run.`}>
              <div className="grid2" style={{ gap: 1, background: "var(--rule)" }}>
                <div style={{ background: "var(--paper)", padding: 14 }}>
                  <div className="mini-label">Attention · top DLA layers</div>
                  {audit.top.attnAttribution.map((r) => (
                    <div className="spread" key={r.layer} style={{ padding: "4px 0", borderBottom: "1px solid var(--rule)" }}>
                      <span className="mono">L{r.layer}</span><span className="mono">{fmt(r.value, 2)}</span>
                    </div>
                  ))}
                  <div className="mini-label" style={{ marginTop: 12 }}>Attention · top ablation layers</div>
                  {audit.top.attnAblation.map((r) => (
                    <div className="spread" key={r.layer} style={{ padding: "4px 0", borderBottom: "1px solid var(--rule)" }}>
                      <span className="mono">L{r.layer}</span><span className="mono">{fmt(r.value, 2)}</span>
                    </div>
                  ))}
                </div>
                <div style={{ background: "var(--paper)", padding: 14 }}>
                  <div className="mini-label">MLP · top DLA layers</div>
                  {audit.top.mlpAttribution.map((r) => (
                    <div className="spread" key={r.layer} style={{ padding: "4px 0", borderBottom: "1px solid var(--rule)" }}>
                      <span className="mono">L{r.layer}</span><span className="mono">{fmt(r.value, 2)}</span>
                    </div>
                  ))}
                  <div className="mini-label" style={{ marginTop: 12 }}>MLP · top ablation layers</div>
                  {audit.top.mlpAblation.map((r) => (
                    <div className="spread" key={r.layer} style={{ padding: "4px 0", borderBottom: "1px solid var(--rule)" }}>
                      <span className="mono">L{r.layer}</span><span className="mono">{fmt(r.value, 2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
            <Prose>
              <p className="aside">{audit.finding}</p>
            </Prose>
          </>
        )}
      </DataState>

      <Panel title={<><b>Figure 18.4</b> · what the atlas can and cannot support</>}
        tight
        note="This is the honesty boundary for the two chapters that follow. A tool being present does not mean the experiment was run; a readout graph is not an ACDC result.">
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr><th>claim</th><th>evidence in this atlas</th><th>status</th></tr>
            </thead>
            <tbody>
              <tr>
                <td className="k">A layer writes toward the answer</td>
                <td>logit lens, direct attribution, block ablation</td>
                <td>measured on V2-Lite</td>
              </tr>
              <tr>
                <td className="k">A head or expert is necessary</td>
                <td>mean ablation and knockout deltas</td>
                <td>measured on V2-Lite</td>
              </tr>
              <tr>
                <td className="k">A specific path carries the behaviour</td>
                <td>no path patching was run</td>
                <td>not claimed</td>
              </tr>
              <tr>
                <td className="k">An ACDC circuit exists for this task</td>
                <td>offline readout pilot in chapter 19; residual-write edge ablations in chapter 23</td>
                <td>causally run · narrow graph</td>
              </tr>
              <tr>
                <td className="k">V4 uses the same mechanism</td>
                <td>configs and published figures only</td>
                <td>not observed</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      <Prose>
        <p>
          The honest summary is that this atlas is strong on <em>localisation</em> and{" "}
          <em>necessity</em>, and deliberately silent on <em>path specificity</em>. That is not
          a defect of the visualisation; it is a statement about the capture. The next chapter
          runs an offline readout-graph pilot, shows where it stops, and explains why a causal
          edge-level ACDC is a larger experiment than the one that produced the pages before it.
          Chapter 23 follows with the real residual-write intervention.
        </p>
      </Prose>

      <Method label="How this chapter is argued">
        <p>
          No new forward passes were run for this chapter. The definitions follow the
          transformer-circuits framing (Elhage et al., 2021), the induction-head result
          (Olsson et al., 2022), the key-value view of MLPs (Geva et al., 2021) and the
          path-patching IOI circuit (Wang et al., 2022). The DeepSeek-specific objects are
          read from the official configs and the V2-Lite capture described in Methods.
          The audit is computed offline by <code>experiments/mechanistic.py</code> from the
          captured <code>dla.json</code> and <code>ablation.json</code>.
        </p>
        <p>
          The ladder is a checklist, not a scoring rubric. A claim can be useful at rung two
          without pretending to be rung four; the failure mode is silently moving up a rung
          when the evidence has not.
        </p>
      </Method>

      <Footnotes items={[
        "Elhage et al., <i>A Mathematical Framework for Transformer Circuits</i>, Anthropic, 2021. transformer-circuits.pub/2021/framework.",
        "Olsson et al., <i>In-context Learning and Induction Heads</i>, arXiv:2209.11895; Geva et al., <i>Transformer Feed-Forward Layers Are Key-Value Memories</i>, arXiv:2012.14913.",
        "Wang et al., <i>Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small</i>, arXiv:2211.00593.",
        "“Component” in this atlas means a hooked attention head, a routed expert, a shared expert, or an MLP block. The router itself is a gate, not a residual writer; its decision is recorded alongside the expert write.",
        "V2-Lite is the only member with captured internals. Every statement about a V4 mechanism in this chapter is a statement about what would need to be measured, not a measurement.",
        "The audit compares mean absolute direct attribution over 27 prompts with absolute mean ablation delta over 12 prompts. The two prompt sets overlap but are not identical; the rank comparison is descriptive.",
      ]} />
      <PageFoot index={18} />
    </>
  );
}
