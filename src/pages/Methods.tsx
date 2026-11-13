import React, { useMemo } from "react";
import { Prompt, useOne, Meta } from "../lib/atlas";
import { ChapterHead, Chip, Footnotes, PageFoot, Panel, Prose, Stat } from "../components/ui";

const SOURCES = [
  { n: "DeepSeek-V2", u: "https://huggingface.co/deepseek-ai/DeepSeek-V2", d: "MLA; 236B total / 21B active; 128k context." },
  { n: "DeepSeek-V2-Lite", u: "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite", d: "The proxy measured here: 15.7B total / 2.4B active, 27 layers, 64 routed + 2 shared experts." },
  { n: "DeepSeek-V3", u: "https://huggingface.co/deepseek-ai/DeepSeek-V3", d: "671B total / 37B active; 256 routed experts, top-8; 61 layers. Technical report: arXiv:2412.19437." },
  { n: "DeepSeek-R1", u: "https://huggingface.co/deepseek-ai/DeepSeek-R1", d: "Reasoning post-training of the V3 base; 671B total / 37B active, 61 layers, 256 routed experts. Architecture is shared with V3." },
  { n: "DeepSeek-V3.2-Exp", u: "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp", d: "Sparse attention via a lightning indexer; 128k context." },
  { n: "DeepSeek-V4-Flash / Pro", u: "https://huggingface.co/deepseek-ai", d: "V4 generation; 284B/13B and 1.6T/49B class configs. V4 architecture notes are from the release cards." },
  { n: "DeepSeek-V4.1 report", u: "https://arxiv.org/abs/2609.19969", d: "Primary source for 890 B/token global KV (FP4, CSA2), 40 layers as 20 causal encoder + 20 decoder, 384 routed + 1 shared expert (6 active), and the 128-token sliding window." },
  { n: "ACDC — Conmy et al.", u: "https://arxiv.org/abs/2304.14997", d: "Edge-level automated circuit discovery; the offline pilot is in chapter 19 and the real residual-write edge search is in chapter 23." },
  { n: "Transformer Circuits — Elhage et al.", u: "https://transformer-circuits.pub/2021/framework/index.html", d: "The residual-stream and QK/OV framing used in chapter 18." },
  { n: "IOI circuit — Wang et al.", u: "https://arxiv.org/abs/2211.00593", d: "Path patching and a hand-discovered circuit; the evidence class this atlas does not attempt." },
  { n: "Goodfire — Already Clean", u: "https://already-clean.goodfire.com/#language-models", d: "The interactive-essay form this atlas borrows: one idea per chapter, a manipulable figure, and the method underneath." },
  { n: "Goodfire Research", u: "https://www.goodfire.com/research", d: "Interpretability framing: mechanisms as objects to be located, not stories to be told." },
];

export default function Methods() {
  const promptsQ = useOne<Prompt[]>("prompts");
  const metaQ = useOne<Meta>("meta");
  const prompts = promptsQ.data ?? [];
  const groups = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of prompts) m.set(p.group, (m.get(p.group) ?? 0) + 1);
    return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
  }, [prompts]);

  return (
    <>
      <ChapterHead n="21" part="III" kicker="Methods"
        title={<>How the numbers were <em>made</em></>}
        standfirst={<>This atlas separates three kinds of claim: arithmetic on official configs, forward passes on one runnable checkpoint, and analytic scaling where no checkpoint can be run. Every figure says which one it is.</>}
        chips={<><Chip tone="blue">reproducibility</Chip><Chip>{metaQ.data?.generatedAt?.slice(0, 10) ?? "2026"}</Chip></>}
      />

      <div className="grid3">
        <Stat k="Measured model" v="V2-Lite" s="15.7B total · 2.4B active" />
        <Stat k="Quantisation" v={metaQ.data?.precision ?? "NF4"} s="4-bit, bf16 compute" />
        <Stat k="Hardware" v="Tesla T4" s="16 GB, Kaggle" />
      </div>

      <Panel title={<><b>Figure 21.1</b> · the pipeline</>}
        note="Two independent paths meet in the frontend. The config path is deterministic and needs no GPU; the capture path is a staged, resumable Kaggle run whose JSON is dropped into public/data. Missing captures fall back to an explicit ‘awaiting capture’ state, never to fabricated values.">
        <div style={{ overflowX: "auto" }}>
          <svg width={760} height={230} style={{ minWidth: 680 }}>
            {[
              { x: 20, y: 24, w: 170, h: 54, t: "official config.json", s: "9 checkpoints" },
              { x: 20, y: 140, w: 170, h: 54, t: "V2-Lite weights", s: "4 safetensors · 31.4 GB" },
              { x: 250, y: 24, w: 170, h: 54, t: "analytic.py", s: "family · budget · cache" },
              { x: 250, y: 140, w: 170, h: 54, t: "suite.py", s: "hooks · 10 stages" },
              { x: 480, y: 82, w: 160, h: 54, t: "public/data/*.json", s: "static artefacts" },
              { x: 680, y: 82, w: 60, h: 54, t: "React", s: "35 views" },
            ].map((b, i) => (
              <g key={i}>
                <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="var(--paper-2)" stroke="var(--rule-2)" />
                <text x={b.x + 10} y={b.y + 22} style={{ fontFamily: "var(--mono)", fontSize: 10, fill: "var(--ink)" }}>{b.t}</text>
                <text x={b.x + 10} y={b.y + 38} style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ink-3)" }}>{b.s}</text>
              </g>
            ))}
            {[
              [190, 51, 250, 51], [190, 167, 250, 167], [420, 51, 480, 100], [420, 167, 480, 118], [640, 109, 680, 109],
            ].map(([x1, y1, x2, y2], i) => (
              <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--rule-2)" strokeWidth={1.2} markerEnd="url(#m)" />
            ))}
            <defs>
              <marker id="m" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--rule-2)" />
              </marker>
            </defs>
            <text x={250} y={214} style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ink-3)" }}>capture stages: lens · attention · router · experts · latent · ablation · probes · context · interventions · edges</text>
          </svg>
        </div>
      </Panel>

      <h2 className="sec">Evidence classes</h2>
      <div className="grid3">
        <Panel title={<><b>A</b> · config arithmetic</>}>
          <p style={{ fontSize: 12, lineHeight: 1.6, color: "var(--ink-2)" }}>
            Parameter counts, active counts, KV bytes and FLOPs are computed from the
            official <code>config.json</code>. No weights are involved. Totals are checked
            against the published parameter counts where available; the residual is
            reported on the Budget page.
          </p>
        </Panel>
        <Panel title={<><b>B</b> · measured on V2-Lite</>}>
          <p style={{ fontSize: 12, lineHeight: 1.6, color: "var(--ink-2)" }}>
            Forward passes on the 4-bit checkpoint: 27 contrast prompts, 96 probe prompts,
            and the smaller causal suites in chapters 22–23. Every measured figure names
            the model, precision and corpus size. A measurement on V2-Lite is evidence
            about V2-Lite, not about V4.
          </p>
        </Panel>
        <Panel title={<><b>C</b> · analytic model</>}>
          <p style={{ fontSize: 12, lineHeight: 1.6, color: "var(--ink-2)" }}>
            Long-context cache and FLOP curves for checkpoints that cannot be run here.
            The model is closed-form and its assumptions are stated. Where an official
            figure exists (V4.1's 890 B/token), it is reported alongside the model and labelled
            with its precision (FP4, 128-token sliding window).
          </p>
        </Panel>
      </div>

      <h2 className="sec">The corpus</h2>
      <Panel title={<><b>Figure 21.2</b> · 27 controlled contrasts</>}
        note="Each prompt is paired with a hand-written continuation and checked to tokenise as a prefix of the concatenation. The corpus is deliberately small and stratified by computation type, not sampled from a benchmark.">
        <div className="grid2" style={{ gap: 18 }}>
          <div className="stack" style={{ gap: 1, background: "var(--rule)" }}>
            {groups.map(([g, n]) => (
              <div key={g} className="spread" style={{ background: "var(--paper)", padding: "7px 10px" }}>
                <span style={{ fontSize: 11.5, color: "var(--ink-2)" }}>{g}</span>
                <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>{n}</span>
              </div>
            ))}
            {!groups.length && <div className="caption" style={{ padding: 10 }}>prompts.json not loaded</div>}
          </div>
          <Prose>
            <p>
              The prompts are not a benchmark. They are contrast pairs chosen so that a
              layer view has something to show: a fact that must be retrieved, a pattern
              that must be continued, a sum that must be carried, a bracket that must be
              closed. The target is always the first token of a continuation that a human
              would accept.
            </p>
            <p>
              The 96 probe prompts are separate: short, labelled utterances whose
              properties (punctuation, digit, code, repetition) are known by construction.
              They are used only for the linear probes.
            </p>
          </Prose>
        </div>
      </Panel>

      <h2 className="sec">Reproducing</h2>
      <Panel title={<><b>Figure 21.3</b> · the commands</>}
        note="The analytic path runs anywhere with Python 3. The capture path needs a GPU with ≥16 GB and a Kaggle account; it is staged so a disconnect resumes rather than restarts.">
        <pre className="code">{`# analytics — no GPU, no weights
python3 experiments/analytic.py
python3 experiments/mechanistic.py   # mechanism.json + acdc.json

# package the capture kernel
python3 experiments/make_kernel.py
cd experiments/kaggle && kaggle kernels push -p .

# frontend
npm install
npm run dev

# after the kernel finishes
kaggle kernels output shariq101/deepseek-atlas-capture -p public/data`}</pre>
        <div className="caption">Outputs are written to <code>public/data/</code>; the app reads them with <code>useJson</code> and shows an explicit empty state if a file is absent.</div>
      </Panel>

      <h2 className="sec">Limitations</h2>
      <Prose>
        <p>
          <strong>One proxy.</strong> All mechanistic claims are V2-Lite. It is the only
          checkpoint in the family that fits a 16 GB T4 at 4-bit. V3 and V4 are analysed
          from configs; their internals were not observed.
        </p>
        <p>
          <strong>Small corpus.</strong> 27 prompts is enough to see a curve, not to
          estimate a distribution. The causal follow-up is smaller still: four prompts for
          the head/expert maps and one discovery plus one validation prompt for the edge
          search. Those denominators are shown on the pages; none of them is a benchmark.
        </p>
        <p>
          <strong>Quantisation.</strong> NF4 perturbs logits and can flip near-ties. The
          layer ordering of the lens is robust; absolute probabilities are not. No
          calibration is applied.
        </p>
        <p>
          <strong>Ablation is not a circuit.</strong> Replacing a sublayer with its mean
          tests necessity, not sufficiency, and the model can compensate. The graph is an
          attribution hypothesis; the ablation is the node-level causal check. Chapter 19
          contains the offline readout-graph pilot. Chapters 22–23 contain the real
          forward-pass head/expert ablations and the causal residual-write edge search.
        </p>
        <p>
          <strong>V4 numbers are published, not measured.</strong> The 890 B/token figure,
          the 40-layer CED and the 384-expert count are from the V4.1 release material.
          They are the best available evidence and are labelled as such.
        </p>
      </Prose>

      <h2 className="sec">Sources</h2>
      <div className="stack" style={{ gap: 1, background: "var(--rule)" }}>
        {SOURCES.map((s, i) => (
          <div key={i} style={{ background: "var(--paper)", padding: "10px 12px", display: "grid", gridTemplateColumns: "170px 1fr", gap: 12 }}>
            <a href={s.u} target="_blank" rel="noreferrer" className="mono" style={{ fontSize: 10, color: "var(--signal)", textDecoration: "none" }}>{s.n} ↗</a>
            <span style={{ fontSize: 11.5, color: "var(--ink-2)", lineHeight: 1.5 }}>{s.d}</span>
          </div>
        ))}
      </div>

      <Footnotes items={[
        `Artefacts generated ${metaQ.data?.generatedAt ?? "—"}; measured runtime ${metaQ.data?.runtimeSec ? Math.round(metaQ.data.runtimeSec / 60) + " min" : "—"}.`,
        "Configs are the official Hugging Face releases, cached in experiments/configs. No values were typed from memory.",
        "The frontend is static: every number on screen comes from a JSON file, not from a component constant.",
        "Where a capture is missing, the chapter says so. The atlas never fills a gap with a plausible-looking number.",
      ]} />
      <PageFoot index={21} />
    </>
  );
}
