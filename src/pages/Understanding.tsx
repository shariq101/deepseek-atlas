import React from "react";
import {
  ChapterHead, Chip, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { ActivationPlayback, ActivationComparison } from "../components/motion";
import { ContextCarryDiagram, SarcasmDiagram, VerbFrameDiagram } from "../components/theory";
import { ContextData, pct, useOne } from "../lib/atlas";

/* --------------------------------------------------------------------------
   Measured retrieval chart. Uses the real context.json capture, not an
   illustrative curve. One synthetic needle prompt per length × depth.
   -------------------------------------------------------------------------- */

function NeedleChart() {
  const state = useOne<ContextData>("context");
  const rows = state.data?.results ?? [];

  if (state.loading) {
    return <div className="u-chart-empty">Loading the measured retrieval capture…</div>;
  }
  if (state.error || !rows.length) {
    return <div className="u-chart-empty">The measured retrieval capture is not available.</div>;
  }

  const lengths = Array.from(new Set(rows.map((r) => r.len))).sort((a, b) => a - b);
  const depths = Array.from(new Set(rows.map((r) => r.depth))).sort((a, b) => a - b);
  const colors = ["var(--signal)", "var(--teal)", "var(--ember)"];
  const W = 660;
  const H = 230;
  const left = 46;
  const right = 16;
  const top = 22;
  const bottom = 42;
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const lo = 0.90;
  const hi = 1.0;
  const x = (i: number) => left + (i / Math.max(1, lengths.length - 1)) * plotW;
  const y = (p: number) => top + (1 - (Math.max(lo, Math.min(hi, p)) - lo) / (hi - lo)) * plotH;

  const point = (len: number, depth: number) =>
    rows.find((r) => r.len === len && Math.abs(r.depth - depth) < 0.001);

  const all = rows.map((r) => r.targetProb);
  return (
    <div className="u-needle">
      <div className="u-needle-legend">
        {depths.map((d, i) => (
          <span key={d}><i style={{ background: colors[i] }} />{Math.round(d * 100)}% depth</span>
        ))}
        <span className="u-needle-range">answer probability · {pct(Math.min(...all))}–{pct(Math.max(...all))}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Measured needle retrieval probability from 256 to 4096 tokens at three depths">
        {[0.90, 0.95, 1.0].map((v) => (
          <g key={v}>
            <line x1={left} y1={y(v)} x2={W - right} y2={y(v)} stroke="var(--rule)" strokeWidth={0.7} strokeDasharray={v === 1 ? undefined : "2 3"} />
            <text x={left - 7} y={y(v) + 2} textAnchor="end" className="chart-endlabel">{v.toFixed(2)}</text>
          </g>
        ))}
        {lengths.map((len, i) => (
          <g key={len}>
            <line x1={x(i)} y1={top} x2={x(i)} y2={H - bottom} stroke="var(--rule)" strokeWidth={0.45} />
            <text x={x(i)} y={H - bottom + 14} textAnchor="middle" className="chart-endlabel">
              {len >= 1024 ? `${Math.round(len / 1024)}k` : len}
            </text>
          </g>
        ))}
        <line x1={left} y1={H - bottom} x2={W - right} y2={H - bottom} stroke="var(--rule-2)" strokeWidth={0.8} />
        {depths.map((d, di) => {
          const pts = lengths
            .map((len, i) => {
              const r = point(len, d);
              return r ? `${x(i)},${y(r.targetProb)}` : null;
            })
            .filter((s): s is string => Boolean(s))
            .join(" ");
          return <polyline key={d} points={pts} fill="none" stroke={colors[di]} strokeWidth={1.6} />;
        })}
        {depths.map((d, di) =>
          lengths.map((len, i) => {
            const r = point(len, d);
            if (!r) return null;
            return (
              <circle key={`${d}-${len}`} cx={x(i)} cy={y(r.targetProb)} r={2.8} fill={colors[di]}>
                <title>{`${len} tokens · ${Math.round(d * 100)}% depth · P(answer)=${r.targetProb.toFixed(3)} · top-1 ${r.top1}`}</title>
              </circle>
            );
          }),
        )}
        <text x={left} y={H - 6} className="chart-endlabel">tokens in context →</text>
        <text x={left - 7} y={15} textAnchor="end" className="chart-endlabel">P(answer)</text>
      </svg>
      <p className="u-chart-caption">
        Every cell in this capture returns the same top-1 answer, <code>7</code>. One synthetic
        key/value needle per length and depth; no sampling. This is a retrieval test, not a
        test of discourse comprehension.
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export default function UnderstandingPage() {
  return (
    <>
      <ChapterHead
        n="25"
        part="V"
        kicker="Theory · meaning and context"
        title={<>How meaning <em>travels</em></>}
        standfirst={
          <>
            A transformer has no dictionary of meanings and no single memory of a paragraph.
            It builds context-dependent vectors, routes information through attention, and
            reconstructs the past from cached states. This chapter separates what the research
            shows from what a fluent answer makes us assume.
          </>
        }
        chips={<><Chip tone="blue">research map</Chip><Chip tone="teal">measured playback</Chip><Chip>long context</Chip></>}
      />

      <div className="grid3" style={{ marginBottom: 22 }}>
        <Stat k="Form" v="Partly probed" s="syntax is decodable, not fully parsed" />
        <Stat k="Sense" v="Contextual" s="a vector changes with its neighbours" />
        <Stat k="Use" v="Hard to prove" s="sarcasm, implication and reasoning" />
      </div>

      <Prose>
        <p className="lede">
          “The model understands” can mean three very different things. It can mean that the
          model represents grammatical structure, that it represents the meaning of a phrase in
          context, or that it uses that meaning in the way a speaker does. These claims need
          different instruments. A grammar benchmark, a linear probe and a causal intervention
          are not interchangeable evidence.
        </p>
        <p>
          The distinction matters because a fluent continuation can make all three look true at
          once. A model can pass an agreement test without carrying a parse tree, answer a factual
          question without storing a proposition, and detect a sarcastic cue without understanding
          the speaker. The rest of this page keeps the evidence class attached to the claim.
        </p>
      </Prose>

      <div className="u-ladder">
        <div>
          <span>01 · FORM</span>
          <b>Is the sentence well formed?</b>
          <p>Agreement, dependencies, phrase structure, word order. This is the easiest level to test with controlled contrasts.</p>
          <small>instruments: acceptability judgments, subject–verb agreement, structural probes</small>
        </div>
        <div>
          <span>02 · SENSE</span>
          <b>What does it mean here?</b>
          <p>Reference, semantic roles, entailment, and the way a word shifts with context. Meaning is relational, not a label attached to a token.</p>
          <small>instruments: semantic-role probes, entailment sets, contextual similarity, causal patching</small>
        </div>
        <div>
          <span>03 · USE</span>
          <b>What is the speaker doing?</b>
          <p>Irony, implicature, stance, reasoning toward a conclusion. The same sentence can do different things in different conversations.</p>
          <small>instruments: pragmatics benchmarks, process checks, faithfulness interventions</small>
        </div>
      </div>

      <Prose>
        <p>
          The strongest evidence for linguistic structure in transformers comes from controlled
          probes, not from attention maps. Hewitt and Manning found that a linear transformation
          of BERT or ELMo representations recovers distances and depths in a parse tree. Jawahar
          and colleagues found surface features in earlier layers and more syntactic information
          in later ones. Rogers and colleagues' survey of BERTology notes that attention weights
          are weak indicators of subject–verb agreement and reflexive anaphora. None of these
          results establishes human-like parsing; each establishes that some linguistic
          information is linearly readable or behaviourally usable.
        </p>
        <p>
          The same caution applies to semantics. Probes can decode semantic roles or properties
          from a representation, and feed-forward layers can be described as key–value memories.
          That tells us information is present and can be routed. It does not tell us that the
          model uses that information the way a person uses a concept. A probe is a correlation;
          an intervention is the step that turns it into a causal claim.
        </p>
      </Prose>

      <h2 className="sec">The measured activation playback</h2>
      <Prose>
        <p>
          This atlas captured one forward pass of V2-Lite on each of 27 prompts. The playback
          defaults to <code>fact-capital</code> — <code>“The capital of Australia is”</code> →
          <code>Canberra</code> — and the prompt selector includes the two negation prompts
          (<code>neg-always</code>, <code>neg-whale</code>) and the two idiom prompts
          (<code>idiom-bucket</code>, <code>idiom-sword</code>). The lens, router and direct
          attribution exist for every captured prompt. Every prompt also has the final query's
          per-head attention row; full T×T matrices are stored for the four deep prompts:
          fact-capital, syn-attract, ind-1 and code-add.
        </p>
        <p>
          The animation synchronises the real readouts for the selected prompt. The lens is read
          at each layer; the attention panel is the final query's per-head map; the
          router panel is the final token's top-6 experts; the attribution is a direct logit
          decomposition for that prompt's target.
        </p>
        <p>
          The lens profile is not monotonic. At the embedding readout the target rank is 70,820.
          It first reaches rank 0 at readout 17, peaks at 99.98% at readout 21, and then falls to
          52.84% at the final readout. That final drop is not a contradiction in the data: an
          intermediate residual read through the final unembedding is not the model's final
          predictive state. Later blocks rewrite the representation. A lens is a probe of
          readability, not a readout of belief.
        </p>
      </Prose>

      <Panel
        title={<><b>Figure 25.1</b> · one prompt, layer by layer</>}
        note="Measured on V2-Lite (NF4). The selector covers all 27 captured prompts and defaults to fact-capital. Every prompt has the final query's per-head attention row; full T×T matrices exist for the four deep prompts. One forward pass per prompt; no sampling. The attention panel shows the final query position only. The router panel shows the final token's selected experts. The attribution is a linear decomposition, not an ablation."
      >
        <ActivationPlayback />
      </Panel>

      <div className="grid3" style={{ marginBottom: 8 }}>
        <Stat k="Readout 0" v="#70,820" s="fact-capital · Canberra's rank" />
        <Stat k="Readout 21" v="99.98%" s="fact-capital · peak lens probability" />
        <Stat k="Final readout" v="52.84%" s="fact-capital · top-1 is still Canberra" />
      </div>

      <Prose>
        <p>
          The attention view answers a narrower question than “what did the model understand?”:
          at the final query, which earlier positions does each head read? The router view answers
          another: which six experts run for the final token at a given block? The attribution view
          answers a third: how much of the target logit is linearly associated with attention and
          MLP writes? None of the four panels is a semantic decoder. Together they show the kind
          of object a meaning claim has to be about: a distributed state, a routing pattern and a
          sequence of writes.
        </p>
        <p className="aside">
          <strong>The useful negative result is in the shape of the curve.</strong> A model can
          make the right answer linearly readable well before its final state, then partially
          overwrite it. Any story that treats one mid-stack probe as “what the model thinks” will
          be too simple.
        </p>
      </Prose>

      <h2 className="sec">Side-by-side readout</h2>
      <Prose>
        <p>
          The comparison keeps <code>fact-capital</code> fixed on the left and lets you choose
          one of the four captured negation or idiom prompts on the right. Both cards read the
          same lens index, so the probabilities, ranks and top predictions are compared at the
          same depth rather than at each prompt's best layer. Both cards show their final-query
          per-head attention rows, so the contrast is not only a probability comparison.
        </p>
      </Prose>

      <Panel
        title={<><b>Comparison</b> · same readout, two prompts</>}
        note="Measured on V2-Lite (NF4). Left is fixed to fact-capital; right is one of neg-always, neg-whale, idiom-bucket or idiom-sword. The shared slider controls both cards. Each card shows the final query's 16-head attention row at the readout layer."
      >
        <ActivationComparison />
      </Panel>

      <h2 className="sec">Semantics: meaning is a relation, not a label</h2>
      <Prose>
        <p>
          A token's representation is a function of its context. <code>bank</code> beside
          <code> river</code> and <code>bank</code> beside <code>loan</code> need not share a
          single static meaning vector; the surrounding tokens change what is represented and what
          can be read out. This is why contextual embeddings replaced static word vectors, and why
          a sentence is not just a bag of token meanings.
        </p>
        <p>
          But “contextual” does not mean “a sentence meaning is stored in one vector”. The residual
          stream at a position is a high-dimensional state that many features can read. Different
          layers can make different properties linearly available. A probe that recovers a semantic
          role from layer 12 shows that the role is decodable there; it does not show that layer 12
          is “where roles live”, or that the model uses that code downstream.
        </p>
        <p>
          Consider: <em>“The curator put the painting in the vault because it was valuable.”</em>{" "}
          A reader resolves <code>it</code> to the painting, not the vault, using syntax, selectional
          preferences and world knowledge. A transformer can exploit the same regularities in its
          context. The linguistic description is useful; the mechanistic claim still needs an
          intervention. Which attention heads bind the pronoun? Which MLP features encode
          “valuable object”? Those are hypotheses, not consequences of the sentence.
        </p>
        <p>
          Factual recall adds another layer. Feed-forward layers can act like key–value memories,
          and causal tracing has localized some factual associations in mid-layer MLPs. Those
          results are model- and task-specific. They do not mean every fact is a row in an MLP, and
          they do not turn a logit-lens readout into a stored proposition.
        </p>
      </Prose>

      <h2 className="sec">Verbs and syntax: a predicate is a frame</h2>
      <Prose>
        <p>
          A verb is not only a label for an event. It selects participants and shapes what the rest
          of the sentence must supply. <code>sleep</code> needs a sleeper. <code>put</code> needs
          something put, a place, and an agent. <code>give</code> needs a giver, a thing given and a
          recipient. Linguists describe this as argument structure or a semantic frame; the exact
          inventory is theory-dependent, but the selectional pattern is real.
        </p>
        <p>
          Transformers have two obvious ways to exploit it. Attention can bind a predicate to its
          arguments across positions, and MLP features can encode selectional preferences such as
          “the thing put is a physical object”. Structural probes show that parse information is
          recoverable from representations; agreement studies show that models track number across
          intervening material. Neither result says the model implements a symbolic frame. It says
          the information is there to be used.
        </p>
      </Prose>

      <Panel
        title={<><b>Figure 25.2</b> · the frame selected by “put”</>}
        note="A linguistic description of argument structure, not a measured circuit. The arrows name the participants the verb licenses; they do not claim that one attention head implements each role."
      >
        <VerbFrameDiagram />
      </Panel>

      <Prose>
        <p>
          Tense and aspect make the problem longer-range. <em>“She was painting the room when the
          phone rang”</em> requires the reader to hold an ongoing activity, a background event and
          their relation. The relevant cues are spread across auxiliaries, participles and
          connectives. This is exactly the kind of dependency that a causal, multi-layer model can
          represent, but it is also where a local n-gram shortcut can look like understanding on a
          small benchmark.
        </p>
        <p>
          Subject–verb agreement is the cleanest test case because meaning and syntax can be
          separated experimentally. Models handle simple agreement well; performance drops with
          attractors and complex structure, and the pattern depends on training and architecture.
          The result is a capability profile, not a claim that the model has a grammar module.
        </p>
      </Prose>

      <h2 className="sec">Sarcasm: the sentence is not the message</h2>
      <Prose>
        <p>
          Sarcasm is a pragmatic inversion. The literal sentence points one way; the speaker's
          attitude points another; the context makes the mismatch visible. Text removes prosody,
          facial expression and gesture, so a text-only model must infer the inversion from words,
          prior turns, speaker identity and world knowledge — or from a dataset shortcut.
        </p>
        <p>
          The iSarcasm dataset was built because previous labelling methods did not capture
          intended sarcasm well. It labels tweets for sarcasm directly by their authors and
          distinguishes intended from perceived sarcasm. MUStARD adds multimodal context from
          television dialogue. Both are useful because they make the contextual dependence explicit;
          neither shows that a transformer has a general theory of irony.
        </p>
      </Prose>

      <div className="u-dialogue">
        <div><span>A</span><p>The client moved the deadline to Friday.</p></div>
        <div><span>B</span><p>Great. Another all-nighter.</p></div>
      </div>

      <Panel
        title={<><b>Figure 25.3</b> · literal sentiment versus speaker meaning</>}
        note="The cue is the mismatch between the utterance and the situation, not the word “great” alone. A model that keys on “great” as positive sentiment will invert the intended meaning."
      >
        <SarcasmDiagram />
      </Panel>

      <Prose>
        <p>
          This is why an attention map over the word <code>great</code> is not evidence of sarcasm
          detection. The relevant computation may involve the deadline, the speaker's prior turns,
          the expected effort, or a contrast between positive wording and negative consequences.
          A benchmark score tells you that a model can classify the dataset; it does not tell you
          which cue it used.
        </p>
        <p>
          The evaluation questions are therefore part of the claim. Whose labels? Speaker intent or
          annotator inference? Same-domain or cross-domain? Is dialogue context included? Does the
          model see the preceding turn or only the target utterance? A result without those details
          is a result about a dataset, not about pragmatic understanding.
        </p>
      </Prose>

      <h2 className="sec">Reasoning: a trace is not a proof</h2>
      <Prose>
        <p>
          Chain-of-thought prompting gives a model intermediate tokens to condition on. On
          multi-step problems this often improves accuracy, and process supervision — rewarding
          each correct step rather than only the final answer — outperforms outcome supervision on
          the MATH benchmark in OpenAI's <em>Let's Verify Step by Step</em>. DeepSeek-R1 uses
          reinforcement learning with verifiable rewards to train long reasoning traces. These are
          real capability results.
        </p>
        <p>
          They do not establish that the written trace is the computation that produced the answer.
          Lanham and colleagues intervened on chain-of-thought steps by introducing mistakes or
          paraphrasing them, and found that models vary widely in how strongly they condition on
          the trace. In their study, larger models were less faithful on most tasks. A plausible
          rationale can be post-hoc; a correct answer can survive removing the stated steps.
        </p>
        <p>
          Keep two questions apart. <strong>Capability:</strong> can the model solve the problem?
          <strong> Mechanism:</strong> did these intermediate states cause the answer? Benchmarks
          answer the first. Process checks, causal patching and faithfulness interventions are
          needed for the second. A reasoning trace is evidence about a model's output distribution,
          not automatically evidence about its internal algorithm.
        </p>
      </Prose>

      <div className="u-proof">
        <div><span>01</span><b>Answer accuracy</b><p>Did it reach the right result?</p><small>behavioural evidence</small></div>
        <div><span>02</span><b>Process accuracy</b><p>Is each step valid and well ordered?</p><small>step-level supervision</small></div>
        <div><span>03</span><b>Causal faithfulness</b><p>Does removing or changing a step change the answer?</p><small>intervention evidence</small></div>
      </div>

      <Prose>
        <p>
          The architectural point is simpler than the debate: reasoning is not a new block. R1
          shares V3's block shape and changes the post-training distribution and objective. If a
          capability difference appears, training and data are the first explanations to check.
          The measured causal pages are the place to ask whether a named component actually carries
          the computation.
        </p>
      </Prose>

      <h2 className="sec">Long context: what is actually carried</h2>
      <Prose>
        <p>
          The most misleading metaphor is a memory file. There is no paragraph vector that gets
          handed from sentence one to sentence eight. At each layer, the current token's query reads
          a cache of earlier keys and values; the weighted read is added to the current residual
          stream. The past is reconstructed at every layer, not forwarded as one summary.
        </p>
      </Prose>

      <div className="u-carry-list">
        <div>
          <span>01 · TOKENS + POSITIONS</span>
          <b>The address survives</b>
          <p>Every token in the window keeps its id and position. RoPE rotates queries and keys by position, so “Mira” in sentence one is still addressable in sentence eight.</p>
        </div>
        <div>
          <span>02 · K/V CACHE</span>
          <b>The substrate survives</b>
          <p>For every earlier token and layer, inference stores a key and a value. V2-Lite's MLA stores a compressed latent and reconstructs per-head K/V. This cache is the exact object a later query reads.</p>
        </div>
        <div>
          <span>03 · ATTENTION</span>
          <b>The read is reconstructed</b>
          <p>A query at the current token scores earlier keys, softmaxes the scores, and reads a weighted sum of values. Different heads and layers can read different relations.</p>
        </div>
        <div>
          <span>04 · RESIDUAL</span>
          <b>Only the present is updated</b>
          <p>The current token's stream accumulates the reads and MLP writes. Earlier tokens are not re-run; generation appends a token and repeats the loop.</p>
        </div>
      </div>

      <Panel
        title={<><b>Figure 25.4</b> · the carry across a paragraph</>}
        note="Conceptual. The cache is per token, per layer, and usually compressed; the diagram shows one read path, not the full attention pattern. There is no single paragraph vector in the residual stream."
      >
        <ContextCarryDiagram />
      </Panel>

      <Prose>
        <p>
          This is why a conversation can feel continuous: the earlier words really are still in
          the window, and each new query can read them. It is also why the continuity is fragile.
          The model does not have a persistent memory of the conversation outside the tokens and
          cached states you give it. If the context is truncated, evicted or summarized, the
          information is gone or lossy. If the weights are frozen, the conversation has not taught
          the model anything permanent.
        </p>
        <p>
          The cache grows with sequence length, layers and stored width. That is the economic
          problem behind long context: attention cost and memory cost. Multi-head latent attention
          compresses what persists; sparse attention reduces which keys are read; CED shares a full
          cache across selected layers. Each is a cost mechanism. None is a guarantee that a
          distant sentence will be understood.
        </p>
      </Prose>

      <h2 className="sec">Why distance still hurts</h2>
      <Prose>
        <p>
          Even inside the window, position is not neutral. <em>Lost in the Middle</em> found that
          retrieval and question answering are often best when the relevant passage is at the
          beginning or end of the context and worst when it is in the middle. More context also
          means more distractors, more chances for interference and a larger cache to compress.
        </p>
        <p>
          Attention maps add a second caution. StreamingLLM showed that a few initial tokens can
          act as attention sinks: a head may put a large weight on them even when they carry little
          task content. Keeping those sinks plus a recent window preserves fluency in streaming
          generation, but it is not the same as retaining every fact. A high attention weight can
          be a no-op, a positional anchor, or a genuine read; the weight alone does not tell you
          which.
        </p>
        <p>
          Position-extension methods stretch the range. Position Interpolation linearly downscales
          position indices so a RoPE model can handle longer sequences with little fine-tuning;
          YaRN is a compute-efficient RoPE-extension method that uses fewer tokens and training
          steps than prior approaches. These methods can make a longer window usable, but they do
          not add knowledge, improve multi-hop reasoning, or guarantee that a middle passage is found.
        </p>
      </Prose>

      <Panel
        title={<><b>Figure 25.5</b> · measured retrieval to 4,096 tokens</>}
        note="V2-Lite · synthetic key/value needle · one prompt per length × depth. The top-1 answer is “7” in every cell. This measures retrieval at a distance, not comprehension of a document."
      >
        <NeedleChart />
      </Panel>

      <Prose>
        <p>
          In this atlas's capture, the answer probability stays between 94.1% and 98.5% from 256
          to 4,096 tokens across the three depths. That is a real result about this setup: a single
          synthetic fact can still be retrieved at 4k. It is not evidence that a model tracks an
          argument across paragraphs. A needle is a local lookup with one relevant span; discourse
          requires reference resolution, temporal order, causal links, and ignoring many plausible
          distractors.
        </p>
        <p>
          This is why “context window” and “memory” are not synonyms. A 128k window is a larger
          address space, not a guarantee that the right address is used. Retrieval quality depends
          on position, interference, the query, the number of hops and the compression scheme.
        </p>
      </Prose>

      <h2 className="sec">Carrying context further</h2>
      <Prose>
        <p>
          If a conversation or document outgrows the window, something must be dropped, compressed
          or fetched. The main options are architectural, not magical:
        </p>
      </Prose>

      <div className="u-options">
        <div>
          <b>Retrieval</b>
          <p>Keep the corpus outside the model, fetch relevant chunks, and put them in the prompt. Explicit and inspectable; retrieval errors become answer errors.</p>
        </div>
        <div>
          <b>Summary / memory tokens</b>
          <p>Compress earlier turns into a summary or a small set of memory tokens. Cheap, but lossy; a summary can omit the detail the next question needs.</p>
        </div>
        <div>
          <b>Sparse / compressed attention</b>
          <p>Read fewer keys, pool keys, share a cache across layers, or compress the KV latent. Reduces cost; the selection and compression rules decide what can be lost.</p>
        </div>
        <div>
          <b>Recurrence / state-space hybrids</b>
          <p>Carry a compressed state across chunks instead of a full cache. Constant memory is attractive; the state is still a bottleneck and can forget.</p>
        </div>
      </div>

      <Prose>
        <p>
          These are directions, not results from this atlas. The measured claim remains narrow:
          this V2-Lite capture retrieves a synthetic needle at 4k. The architectural chapters give
          the cost arithmetic; the causal chapters give the rule for turning a correlation into a
          mechanism claim. None of them shows that a long context is understood the way a person
          understands a conversation.
        </p>
      </Prose>

      <h2 className="sec">What the evidence supports</h2>

      <div className="u-claims">
        <div className="u-claim-head"><span>Claim</span><span>Evidence</span><span>Limit</span></div>
        <div><b>Some syntax is encoded in vector geometry</b><span>Structural probes recover parse distance; agreement tests show sensitivity to attractors.</span><em>Probes are correlational; a recoverable tree is not a used parser.</em></div>
        <div><b>Meaning is context-dependent</b><span>Contextual embeddings and semantic-role probes outperform static-token views.</span><em>No complete semantic code; a probe can decode what the model does not use.</em></div>
        <div><b>Sarcasm requires context and world knowledge</b><span>Datasets with speaker intent and multimodal dialogue expose the dependence.</span><em>Benchmark accuracy can come from cues and shortcuts; intended and perceived labels can differ.</em></div>
        <div><b>Reasoning traces can help capability</b><span>Process supervision and RL with verifiable rewards improve multi-step performance.</span><em>A generated trace is not automatically a faithful explanation.</em></div>
        <div><b>Long context is carried by cached states and attention</b><span>KV cache, RoPE and attention are the architectural mechanism; the atlas retrieves a needle at 4k.</span><em>Retrieval is not discourse understanding; position and compression still hurt.</em></div>
      </div>

      <Prose>
        <p className="aside">
          <strong>The honest summary.</strong> A transformer can use meaning-like structure without
          possessing a human-like semantic store. Its context is reconstructed from tokens, cached
          keys and values, and attention at each layer. Long context is bounded, lossy and
          position-sensitive even when it works. The right question is not “does it understand?”
          in the abstract, but “which information, at which layer, is used for which behaviour,
          and what intervention would show it?”
        </p>
      </Prose>

      <Method label="What is measured and what is not">
        <p>
          The activation playback is one forward pass of V2-Lite (NF4) per selected prompt, with
          <code>fact-capital</code> as the default. The lens is the intermediate residual read
          through the final unembedding; the attention panel is the final query's per-head weights;
          the router panel is the final token's selected experts; the attribution is a direct logit
          decomposition. The comparison section reuses the same captures at one shared readout:
          fact-capital on the left, one negation or idiom prompt on the right. The retrieval
          chart is the existing synthetic-needle capture at 256–4,096 tokens. No R1 forward pass, no
          long-document comprehension run, and no human semantic annotation are included. Literature
          claims are cited below; they are not measurements made by this atlas.
        </p>
      </Method>

      <Footnotes
        items={[
          <><a href="https://aclanthology.org/2020.tacl-1.54.pdf" target="_blank" rel="noreferrer">Rogers, Kovaleva &amp; Rumshisky, “A Primer in BERTology: What We Know About How BERT Works”</a>, TACL 2020. Survey of over 150 BERT studies; notes that attention weights are weak indicators of agreement and anaphora.</>,
          <><a href="https://aclanthology.org/N19-1419/" target="_blank" rel="noreferrer">Hewitt &amp; Manning, “A Structural Probe for Finding Syntax in Word Representations”</a>, NAACL 2019. Parse-tree distance and depth are recoverable under a linear transformation.</>,
          <><a href="https://aclanthology.org/P19-1356.pdf" target="_blank" rel="noreferrer">Jawahar, Sagot &amp; Seddah, “What Does BERT Learn about the Structure of Language?”</a>, ACL 2019. Surface features in earlier layers; harder agreement cases need deeper layers.</>,
          <><a href="https://aclanthology.org/Q16-1037/" target="_blank" rel="noreferrer">Linzen, Dupoux &amp; Goldberg, “Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies”</a>, TACL 2016. The agreement-with-attractors paradigm used to test syntactic sensitivity.</>,
          <><a href="https://aclanthology.org/2021.emnlp-main.446/" target="_blank" rel="noreferrer">Geva et al., “Transformer Feed-Forward Layers Are Key-Value Memories”</a>, EMNLP 2021. A useful description of MLP computation; not a claim that every fact is a stored row.</>,
          <><a href="https://arxiv.org/abs/2202.05262" target="_blank" rel="noreferrer">Meng et al., “Locating and Editing Factual Associations in GPT”</a>, NeurIPS 2022. Causal tracing localizes some factual recalls; results are model- and task-specific.</>,
          <><a href="https://aclanthology.org/J05-1004/" target="_blank" rel="noreferrer">Palmer, Gildea &amp; Kingsbury, “The Proposition Bank: An Annotated Corpus of Semantic Roles”</a>, Computational Linguistics 2005. Semantic-role annotation for predicate–argument structure.</>,
          <><a href="https://aclanthology.org/2020.acl-main.118/" target="_blank" rel="noreferrer">Oprea &amp; Magdy, “iSarcasm: A Dataset of Intended Sarcasm”</a>, ACL 2020. Tweets labelled by their authors; distinguishes intended from perceived sarcasm and shows limits of prior labelling.</>,
          <><a href="https://aclanthology.org/P19-1455/" target="_blank" rel="noreferrer">Castro et al., “Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)”</a>, ACL 2019. MUStARD: audiovisual utterances with dialogue context from television.</>,
          <><a href="https://arxiv.org/abs/2305.20050" target="_blank" rel="noreferrer">Lightman et al., “Let's Verify Step by Step”</a>, ICLR 2024. Process supervision outperforms outcome supervision on MATH.</>,
          <><a href="https://arxiv.org/abs/2307.13702" target="_blank" rel="noreferrer">Lanham et al., “Measuring Faithfulness in Chain-of-Thought Reasoning”</a>, 2023. Interventions on CoT steps show that stated reasoning is not always faithful.</>,
          <><a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”</a>, 2025. Reinforcement learning with verifiable rewards for long reasoning traces.</>,
          <><a href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noreferrer">Liu et al., “Lost in the Middle”</a>, TACL 2024. Retrieval quality varies with the position of relevant information in a long context.</>,
          <><a href="https://arxiv.org/abs/2309.17453" target="_blank" rel="noreferrer">Xiao et al., “Efficient Streaming Language Models with Attention Sinks”</a>, ICLR 2024. Initial-token sinks and a recent window support streaming; not full recall.</>,
          <><a href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">Chen et al., “Extending Context Window of Large Language Models via Positional Interpolation”</a>, 2023. Linear position scaling extends RoPE context with limited fine-tuning.</>,
          <><a href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">Peng et al., “YaRN: Efficient Context Window Extension of Large Language Models”</a>, 2023. Compute-efficient RoPE context extension with fewer training tokens and steps.</>,
          <><a href="https://arxiv.org/abs/2306.14048" target="_blank" rel="noreferrer">Zhang et al., “H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models”</a>, NeurIPS 2023. Heavy-hitter KV eviction trades memory for recall.</>,
          <><a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads" target="_blank" rel="noreferrer">Olsson et al., “In-context Learning and Induction Heads”</a>, 2022. Evidence that induction heads can copy local patterns; causal for small attention-only models, correlational for larger ones.</>,
          <><a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noreferrer">DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model”</a>, 2024. MLA's compressed KV latent, the cache substrate used by the measured model.</>,
          <>The V2-Lite capture in this atlas uses NF4 weights and a single forward pass per prompt. The lens, attention, router and attribution values shown in Figure 25.1 come from those prompt captures; every prompt has the final-query attention row, while full T×T matrices are stored for the four deep prompts. The 256–4,096 retrieval values in Figure 25.5 come from the separate context capture. Neither is a human semantic judgement.</>,
          <>Citation audit, 13 November 2026: all 19 references above were checked against their publisher, ACL Anthology, arXiv or Transformer Circuits record and against the claim they support. The MUStARD link was corrected to P19-1455; the Position Interpolation and YaRN titles were completed; the iSarcasm description was narrowed to intended-versus-perceived labels; and the YaRN claim was narrowed to its reported training efficiency.</>,
        ]}
      />

      <PageFoot index={25} />
    </>
  );
}
