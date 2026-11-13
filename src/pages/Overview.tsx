import React from "react";
import { CHAPTERS, Meta, href, useJson } from "../lib/atlas";
import RoutingField from "../components/RoutingField";
import { Caption, Chip, Footnotes, PageFoot, Panel, Prose, Stat } from "../components/ui";

export default function Overview() {
  const { data } = useJson<Meta>("meta");
  const meta = data?.meta;

  return (
    <>
      <header className="hero">
        <div className="hero-top">
          <div>
            <div className="kicker">A measurement report · November 2026</div>
            <h1>
              Twenty-two ways to read a <em>sparse</em> mind
            </h1>
            <p className="lead">
              The DeepSeek family is the clearest large-scale answer to a question
              the field keeps asking: what if knowledge were stored in experts,
              and only a few of them woke for each token? This atlas measures the
              idea from the only member small enough to instrument —{" "}
              <b>V2-Lite, 15.7B parameters, 2.4B active</b> — and computes the
              rest of the lineage exactly from its published configs: 236B, 671B, 284B,
              1.6T, 552B.
            </p>
            <div className="byline">
              <Chip tone="blue">9 checkpoints</Chip>
              <Chip>36 chapters</Chip>
              <Chip tone="teal">{meta ? `${meta.corpus} measured prompts` : "27 controlled contrasts"}</Chip>
              <Chip>1M context · config</Chip>
            </div>
          </div>
          <div className="hero-side">
            <Stat k="Sparsity, V2-Lite" v="6.4×" s="15.7B total · 2.45B computed active" />
            <Stat k="Sparsity, V4.1" v="32.5×" s="552B total · 17B computed active" />
            <Stat k="KV latent" v="576" s="channels per token, per layer" />
            <Stat k="Measured depth" v={meta ? meta.layers : "27"} s="blocks, every one hookable" />
          </div>
        </div>
        <RoutingField />
      </header>

      <h2 className="sec">The argument</h2>
      <Prose>
        <p>
          A dense transformer gives the interpretability researcher a stable
          object: one residual stream, one MLP per block, every parameter used by
          every token. A sparse MoE breaks that stability on purpose. At each
          block the token is <em>routed</em>: a small gate chooses a few of 64,
          256 or 384 experts, and the rest of the parameters do not run. The unit
          of computation is no longer the neuron. It is the <strong>decision</strong> —
          which experts, how strongly, and what the always-on shared expert adds
          regardless.
        </p>
        <p>
          That decision is measurable, and it is remarkably cheap to measure: one
          forward pass exposes the gate's scores, the selected experts, the
          attention rows and the residual stream at every depth. This report uses
          thirty-six chapters on the same short corpus, chosen as controlled
          contrasts rather than random text: factual recall, agreement under
          attraction, induction, arithmetic, code, multilingual lookup, negation,
          idiom and retrieval.
        </p>
        <p>
          The result is not a tour of one checkpoint. It is an argument about a
          family: <strong>the routing layer is where scale is spent, and the
          residual stream is where it is spent well.</strong> The measured member
          is small enough to see every head and every expert; the config arithmetic
          carries those measurements to the 1.6T flagship without pretending that
          a 15.7B model <em>is</em> a 1.6T model.
        </p>
      </Prose>

      <div className="grid3">
        <Stat k="Total parameters" v="15.7B → 1.6T" s="nine checkpoints, one grammar" />
        <Stat k="Active per token" v="2.4B → 49B" s="the number that sets inference cost" />
        <Stat k="Experts per layer" v="64 → 384" s="routed; plus one shared expert" />
      </div>

      <h2 className="sec">The evidence contract</h2>
      <Prose>
        <p>
          Every figure on this site states which of three kinds of claim it is
          making. The distinction is not decoration: it is the difference between
          a measurement, a derivation and a guess.
        </p>
      </Prose>
      <Panel
        title={<><b>Figure 0.1</b> · what each number is allowed to mean</>}
        tight
        note="The relative error between computed and published counts is shown on the family page. For V2-Lite, V3 and V4.1 the total counts reproduce within 0.2%; active counts differ by about 1–6%, because the configs do not expose every fused or draft parameter."
      >
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr><th>class</th><th>source</th><th>example</th><th>what it can support</th></tr>
            </thead>
            <tbody>
              <tr>
                <td className="k">config</td>
                <td>official config.json</td>
                <td>1.6T total, 49B active, 61 layers, 384 experts</td>
                <td>architecture, parameter budget, KV arithmetic, attention cost</td>
              </tr>
              <tr>
                <td className="k">measured</td>
                <td>forward passes on V2-Lite</td>
                <td>router entropy, logit lens, expert load, ablation ΔNLL, head/expert and edge ablations</td>
                <td>what this checkpoint does on this corpus, not what the family does</td>
              </tr>
              <tr>
                <td className="k">model</td>
                <td>explicit formula, shown</td>
                <td>KV bytes at 1M tokens; attention FLOPs under CSA</td>
                <td>scaling intuition when the checkpoint cannot be run</td>
              </tr>
              <tr>
                <td className="k">not claimed</td>
                <td>—</td>
                <td>“expert 37 means chemistry”; “attention explains the answer”</td>
                <td>no complete path-specific answer yet; the causal pages test components, not semantic labels</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      <h2 className="sec">Twenty-two instruments</h2>
      <div className="cards">
        {CHAPTERS.slice(1).map((c) => (
          <a className="card" key={c.slug} href={href(c.slug)}>
            <span className="n">{c.n}</span>
            <span className="t">{c.title}</span>
            <span className="d">{c.desc}</span>
          </a>
        ))}
      </div>

      <h2 className="sec">Why this family</h2>
      <Prose>
        <p>
          Three architectural bets recur across every DeepSeek generation.{" "}
          <strong>Multi-head Latent Attention</strong> compresses keys and values
          into a low-rank latent, so the cache grows with the latent width rather
          than with heads × head-dim. <strong>DeepSeekMoE</strong> splits the FFN
          into many routed experts and one shared expert, so total capacity can
          grow while active compute stays small. <strong>Multi-token prediction</strong>{" "}
          adds draft heads that predict beyond the next token, turning spare
          compute into speculative speed.
        </p>
        <p>
          V4 adds two more: <strong>hash routing</strong> in the first three MoE
          layers, where the expert is a fixed function of the token id, and a{" "}
          <strong>hybrid compressed/sparse attention</strong> that treats long
          context as a compression problem rather than a quadratic one. V4.1 goes
          further, sharing KV across layers and splitting the model into a causal
          encoder and decoder. Each of those is a chapter here, because each
          changes what a “layer” is.
        </p>
      </Prose>

      <Caption>
        The atlas is static: HTML, CSS, a little JavaScript and the JSON written by
        the capture suite. No model is downloaded to read it. The one measured
        checkpoint shipped here is DeepSeek-V2-Lite, quantised to NF4 for a single
        free GPU; the precision is printed on every measured page because
        quantisation is part of the instrument, not a detail to hide.
      </Caption>

      <Footnotes
        items={[
          "DeepSeek-V4 and V4.1 are recent enough that published summaries disagree on small details; every config number on this site is read from the checkpoint's own <code>config.json</code>, and the source URL is listed on the family page.",
          "The measured corpus is 27 prompts, 313 tokens. It is deliberately small: the point is dense instrumentation of the same computations, not benchmark coverage. Every measured claim is conditional on it.",
          "Parameter counts are computed from named tensor dimensions. V4's grouped low-rank output projection and V4.1's cross-layer cache are the only places where the config does not name every factor; both are flagged where they appear.",
          "This is an independent report. It is not affiliated with DeepSeek AI; the checkpoints and papers are theirs, the instruments and any errors are ours.",
        ]}
      />
      <PageFoot index={0} />
    </>
  );
}
