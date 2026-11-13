import React from "react";
import { useOne, Family } from "../lib/atlas";
import { ChapterHead, Chip, Footnotes, PageFoot, Panel, Prose, Stat } from "../components/ui";

const CLAIMS: { claim: string; evidence: string; grade: "A" | "B" | "C"; note: string }[] = [
  {
    claim: "Sparsity, not depth, is the family's scaling axis.",
    evidence: "Config arithmetic across 9 checkpoints",
    grade: "A",
    note: "Stored parameters grow 100× from V2-Lite to V4-Pro; active parameters grow ~20×. The ratio is the design.",
  },
  {
    claim: "MLA is the reason long context is affordable.",
    evidence: "KV arithmetic + V2-Lite measured cache",
    grade: "A",
    note: "30.4 KiB/token at fp16 for V2-Lite, 7.1× below MHA; the gap widens with head count.",
  },
  {
    claim: "Routing is input-dependent and reproducible, but not a topic taxonomy.",
    evidence: "V2-Lite router capture, 27 prompts",
    grade: "B",
    note: "Co-activation blocks are stable in this corpus; per-expert ‘topics’ have small margins and are not a taxonomy. A corpus swap was not run.",
  },
  {
    claim: "The answer becomes legible before it becomes probable.",
    evidence: "Logit lens, 27 prompts",
    grade: "B",
    note: "Targets can remain far below the top of the distribution for many layers, then collapse to argmax; the gap is the model's deliberation.",
  },
  {
    claim: "A late MLP is causally necessary; early blocks are not uniformly replaceable.",
    evidence: "Ablation sweep",
    grade: "B",
    note: "The largest positive knockout is L26 MLP; L0 is an outlier with the opposite sign, so the depth story is not monotone.",
  },
  {
    claim: "V4.1's 890 B/token cache is a regime change.",
    evidence: "arXiv 2609.19969 + official model card",
    grade: "A",
    note: "A 128-token sliding window plus sparse attention changes what ‘long context’ costs; the figure is published, not measured here.",
  },
  {
    claim: "The family is a single architecture lineage with three substitutions.",
    evidence: "Configs + release notes",
    grade: "A",
    note: "MLA from V2, fine-grained MoE from V3, compressed attention from V3.2/V4. The constants move; the skeleton does not.",
  },
];

const GRADE = {
  A: { label: "config / published", color: "var(--signal)" },
  B: { label: "measured proxy", color: "var(--ember)" },
  C: { label: "analytic model", color: "var(--teal)" },
};

export default function Synthesis() {
  const famQ = useOne<Family>("family");
  const fam = famQ.data;

  return (
    <>
      <ChapterHead n="20" part="III" kicker="Synthesis"
        title={<>One architecture, <em>three substitutions</em></>}
        standfirst={<>The DeepSeek family is often read as a sequence of bigger models. The configs tell a more specific story: a stable MLA + fine-grained MoE skeleton, scaled by moving three constants — expert count, active width, and the attention compression scheme.</>}
        chips={<><Chip tone="blue">synthesis</Chip>          <Chip>{fam?.members.length ?? 9} checkpoints</Chip><Chip>evidence-graded</Chip></>}
      />

      <div className="grid3">
        <Stat k="Stored parameters" v="0.016 → 1.6T" s="V2-Lite → V4-Pro" />
        <Stat k="Active parameters" v="2.4 → 49B" s="~20×" />
        <Stat k="KV per token" v="30.4 KiB → 890 B" s="fp16 MLA → fp4 CSA2" />
      </div>

      <Prose>
        <p>
          Read the nine checkpoints as one design space and the axes are unusually
          clear. <strong>Depth</strong> barely moves: 27 to 61 layers.{" "}
          <strong>Width</strong> moves more, but not by orders of magnitude. What moves by
          orders of magnitude is the number of stored experts — 64, 160, 256, 384 — and
          the compression of the attention state. The family is not scaling a dense
          transformer. It is scaling a sparse memory and a cheap read.
        </p>
        <p>
          <strong>R1 is the control for that reading.</strong> It keeps V3's block exactly: 61
          layers, 256 routed experts, top-8 routing and one shared expert. What changes is the
          post-training objective and the reasoning traces, not the architecture. That makes
          R1 useful as a natural comparison: if a capability differs between V3 and R1 while
          the block is fixed, the difference is a training story before it is an architecture
          story. This atlas has not captured R1's internals, so it states the control, not the
          outcome.
        </p>
        <p>
          That choice has a consequence for every layer view in this atlas. A dense model's
          MLP is a single function; a V2-Lite MoE block is 64 functions of which six run.
          A dense model's KV cache is a fixed cost per head; MLA's is a 512-dimensional
          latent plus a rope key. The familiar pictures — the residual stream, the logit
          lens, the attention map — still apply, but the objects they describe are
          different. The measurements in Part II are the V2-Lite version of those objects;
          the configs in Part I are how they change in V4.
        </p>
      </Prose>

      <Panel title={<><b>Figure 20.1</b> · the evidence ladder</>}
        note="Every claim in this atlas carries its provenance. The ladder is deliberately short at the top: only config arithmetic and published figures are treated as established. Measured claims come from one 15.7B proxy; analytic claims are models, not observations.">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 1, background: "var(--rule)" }}>
          {(["A", "B", "C"] as const).map((g) => (
            <div key={g} style={{ background: "var(--paper)", padding: "18px 18px 14px" }}>
              <div className="row" style={{ gap: 8, marginBottom: 8 }}>
                <span style={{ width: 9, height: 9, background: GRADE[g].color, display: "inline-block" }} />
                <span className="mono" style={{ fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>{GRADE[g].label}</span>
              </div>
              <div className="serif" style={{ fontSize: 26, lineHeight: 1 }}>{CLAIMS.filter((c) => c.grade === g).length}</div>
              <div className="caption" style={{ marginTop: 5 }}>
                {g === "A" && "Official configs and model cards. Arithmetic checked against published totals."}
                {g === "B" && "Forward passes on V2-Lite, NF4, 27 prompts. Directionally robust, sample-limited."}
                {g === "C" && "Closed-form scaling of cache and FLOPs. No weights were run at this scale."}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <h2 className="sec">The claims</h2>
      <div className="stack" style={{ gap: 1, background: "var(--rule)" }}>
        {CLAIMS.map((c, i) => (
          <div key={i} style={{ background: "var(--paper)", padding: "14px 16px", display: "grid", gridTemplateColumns: "22px 1fr 150px", gap: 12 }}>
            <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)", paddingTop: 2 }}>{String(i + 1).padStart(2, "0")}</span>
            <div>
              <div className="serif" style={{ fontSize: 17, lineHeight: 1.3 }}>{c.claim}</div>
              <div className="caption" style={{ marginTop: 3 }}>{c.note}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <span className="mono" style={{ fontSize: 8.5, letterSpacing: "0.06em", textTransform: "uppercase", color: GRADE[c.grade].color }}>● {GRADE[c.grade].label}</span>
              <div className="caption" style={{ marginTop: 3 }}>{c.evidence}</div>
            </div>
          </div>
        ))}
      </div>

      <h2 className="sec">What this atlas does not show</h2>
      <Prose>
        <p>
          It does not show V4's internals. No V4 checkpoint was run; every V4 layer count,
          expert count and cache figure is read from a config or a model card. The
          mechanistic views are V2-Lite's, and V2-Lite is a small model from a previous
          generation. The honest reading is <em>this is how the family's architecture
          behaves at the only scale we can open</em>, not <em>this is what V4 does</em>.
        </p>
        <p>
          It also does not show a circuit in the strong sense. The graph in chapter 14 is
          an attribution hypothesis; the ablation in chapter 13 is a causal intervention.
          Where they disagree, the ablation wins. A real circuit claim would need
          path-specific patching and a behaviour that survives it, which is beyond a
          single-T4 capture. Chapter 19 runs an offline readout pilot; chapter 23 runs the
          real residual-write edge search, with discovery and held-out validation prompts.
          It is still narrower than token-level path patching, so the strong circuit claim
          remains open.
        </p>
      </Prose>

      <Panel title={<><b>Figure 20.2</b> · the three substitutions, plus the control branch</>}
        note="The lineage in one line each. The constants are the story: expert count up, active width up more slowly, attention state down. R1 is the same architecture as V3; it is included to separate post-training from shape.">
        <div className="stack" style={{ gap: 14 }}>
          {[
            { y: "V2 → V3", t: "MLA + fine-grained MoE", d: "V2: 160 routed, top-6, 2 shared, 236B/21B. V3: 256 routed, top-8, 1 shared, 671B/37B. The skeleton is fixed." },
            { y: "V3 → V3.2", t: "sparse attention", d: "A lightning indexer selects top-2048 keys; dense full attention becomes a sparse read." },
            { y: "V3.2 → V4", t: "compressed latent + CED", d: "CSA2 / HCA compress the cache; V4.1 reports 890 B/token at fp4 with a 128-token sliding window." },
            { y: "V3 → R1", t: "reasoning post-training", d: "Same 671B/37B V3 block: 61 layers, 256 routed experts, top-8, one shared. R1 changes training and reasoning traces, not the architecture — a control branch, not a fourth substitution." },
          ].map((r, i) => (
            <div key={i} style={{ display: "grid", gridTemplateColumns: "88px 190px 1fr", gap: 12, alignItems: "baseline" }}>
              <span className="mono" style={{ fontSize: 10, color: "var(--ember)" }}>{r.y}</span>
              <span className="serif" style={{ fontSize: 16 }}>{r.t}</span>
              <span style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.5 }}>{r.d}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Prose>
        <p className="lede">
          The family's bet is that intelligence at the frontier is a problem of{" "}
          <em>memory organisation</em>, not raw width. Every layer view here is an attempt
          to see that organisation: which experts hold what, which blocks write the answer,
          which dimensions the cache keeps, and what survives when a component is removed.
          The pictures are not the model. They are the model's accounting, made visible.
        </p>
      </Prose>

      <Footnotes items={[
        "Claim grades are editorial, not statistical. A grade-A claim can still be wrong if a config is wrong; the configs are the official Hugging Face releases.",
        "Measured claims use 27 prompts. That is enough to see structure, not enough to estimate a population.",
        "R1's architectural constants are read from the official family config; no R1 forward pass or internal capture exists in this atlas.",
        "The V4.1 cache figure is at fp4 with a 128-token sliding window; it is not directly comparable to the fp16 V2-Lite figure without adjusting precision.",
        "All URLs and versions are in the Methods chapter.",
      ]} />
      <PageFoot index={20} />
    </>
  );
}
