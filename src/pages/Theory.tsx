import React, { useState } from "react";
import {
  ChapterHead, Chip, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import {
  AttentionDiagram, CedDiagram, KeywordMap, KeywordPipeline, LatentDiagram, MoeDiagram,
  MultiHeadDiagram, ReasoningDiagram, SelfAttentionMask, SparseDiagram, TransformerDiagram,
} from "../components/theory";
import { TaskAnimation, TransformerPlayback } from "../components/motion";
import type { TaskStep } from "../components/motion";

const ANIMATION_NOTE =
  "A conceptual animation of the task, not a timed model run. The stages and magnitudes are illustrative; the measured pages remain the evidence.";

const TASKS: Record<string, TaskStep[]> = {
  attention: [
    { label: "query", detail: "the current position asks what it needs", gain: "a question in key space" },
    { label: "scores", detail: "q · k for every visible position", gain: "a relevance score per key" },
    { label: "softmax", detail: "scores become a sum-to-one distribution", gain: "a soft comparison, not a lookup" },
    { label: "read + write", detail: "weighted values pass through W_O", gain: "one residual write from the sequence" },
  ],
  self: [
    { label: "emit q/k/v", detail: "every position projects itself into query, key and value", gain: "a parallel read interface for the whole sequence" },
    { label: "mask the future", detail: "position t may read only positions ≤ t", gain: "causality without recurrence" },
    { label: "weight the past", detail: "softmax distributes attention over visible keys", gain: "a context-dependent read" },
    { label: "write back", detail: "the weighted values return to the residual stream", gain: "a token update that uses other positions" },
  ],
  multi: [
    { label: "split", detail: "the hidden state is projected into several head subspaces", gain: "several independent read/write channels" },
    { label: "QK per head", detail: "each head scores keys with its own map", gain: "different relations can be selected in parallel" },
    { label: "OV per head", detail: "each head reads its own weighted values", gain: "different content can be written per relation" },
    { label: "concat + W_O", detail: "heads are concatenated and mixed into one residual write", gain: "parallel subspaces become one block write" },
  ],
  moe: [
    { label: "route", detail: "a gate scores every routed expert for this token", gain: "input-dependent capacity selection" },
    { label: "top-k", detail: "only the k best experts are selected", gain: "sparse compute from a much larger store" },
    { label: "expert compute", detail: "the selected SwiGLU experts transform the token", gain: "specialised feature transforms" },
    { label: "combine", detail: "weighted routed outputs are added to the shared expert", gain: "shared computation plus specialised capacity" },
  ],
  sparse: [
    { label: "index", detail: "a lightweight indexer estimates which keys matter", gain: "a cheap selection signal" },
    { label: "select", detail: "the query keeps a small key set or a pooled view", gain: "a smaller attention matrix" },
    { label: "local window", detail: "recent tokens stay exact and dense", gain: "fine local resolution at the frontier" },
    { label: "read", detail: "the head reads the selected keys and writes the result", gain: "long-context attention at a bounded budget" },
  ],
  latent: [
    { label: "compress", detail: "the token state is projected into a low-rank latent", gain: "a much smaller persistent state" },
    { label: "store", detail: "only the latent plus RoPE channels enter the cache", gain: "less memory per token and layer" },
    { label: "reconstruct", detail: "per-head keys and values are rebuilt on demand", gain: "full-rank reads from a compressed cache" },
    { label: "attend", detail: "the heads use the reconstructed K/V without changing the block", gain: "cache savings without a new attention rule" },
  ],
  ced: [
    { label: "encode prefix", detail: "a causal encoder reads the input context", gain: "a full representation of the prefix" },
    { label: "full cache", detail: "only a few layers compute the expensive global cache", gain: "expensive state is paid sparsely" },
    { label: "share", detail: "the decoder layers read the shared cache", gain: "context reuse without duplicating the store" },
    { label: "decode locally", detail: "recent tokens stay in a local sliding window", gain: "causal generation with a smaller working set" },
  ],
  reasoning: [
    { label: "sample", detail: "the policy produces long reasoning traces", gain: "intermediate steps as training data" },
    { label: "verify", detail: "a verifier checks the final answer or process", gain: "a signal beyond the next-token label" },
    { label: "reward", detail: "verifiable outcomes become a reward", gain: "an objective for useful trajectories" },
    { label: "update", detail: "RL updates the same transformer block", gain: "a capability change without an architecture change" },
  ],
};

const SOURCE_NOTE = (
  <>
    These pages are conceptual. The DeepSeek-specific counts named on them are from the
    official configs; the measured behaviour lives in Parts II and IV. A diagram is not a
    measurement, and a label is not a mechanism.
  </>
);

function TheoryFoot({ index }: { index: number }) {
  return <PageFoot index={index} />;
}

export function TransformerPage() {
  return (
    <>
      <ChapterHead n="24" part="V" kicker="Theory · the whole stack"
        title={<>The transformer, <em>from token to logit</em></>}
        standfirst={<>A transformer is not a pile of named boxes. It is a residual stream that is read and written by a fixed sequence of attention and MLP sublayers, then projected into a vocabulary. This page follows one forward pass from text to probability.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>one forward pass</Chip><Chip tone="teal">residual stream</Chip></>}
      />

      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="The loop" v="Depth" s="not recurrence over time" />
        <Stat k="The bus" v="Residual stream" s="every sublayer reads and writes it" />
        <Stat k="The output" v="Logits → softmax" s="one distribution per position" />
      </div>

      <Prose>
        <p className="lede">
          At inference the model receives a sequence of token ids. Each id becomes an
          embedding vector. A stack of identical blocks transforms every vector in parallel;
          each block reads the current stream through attention, adds what it read, then
          reads the result through a feed-forward network and adds that too. The final
          vector at the last position is normalised and multiplied by the unembedding
          matrix to produce one logit per vocabulary item.
        </p>
        <p>
          The phrase “identical blocks” hides the important part: the blocks are identical
          in <em>shape</em>, not in <em>function</em>. Early blocks tend to build local
          features, middle blocks compose them, and late blocks write toward the answer.
          The residual stream is what makes that composition possible: every block can read
          everything earlier blocks wrote without passing it through a bottleneck.
        </p>
      </Prose>

      <Panel title={<><b>Figure 24.1</b> · one forward pass</>}
        note="Schematic. The block count is the model's config; V2-Lite uses 27 blocks. Attention and the MLP are parallel branches in the diagram only for legibility; in the block they are sequential residual writes.">
        <TransformerDiagram />
      </Panel>

      <Panel title={<><b>Figure 24.2</b> · play the whole pass</>}
        note="An illustrative walk from tokens to a next-token distribution. The linguistic roles are a teaching gloss, not a measured feature readout; the measured forward-pass evidence is in Parts II and IV.">
        <TransformerPlayback />
      </Panel>

      <Prose>
        <p className="aside">
          <strong>A measured version.</strong> Chapter 25 replays a real V2-Lite capture of
          “The capital of Australia is”: the lens readout, the actual attention map, the router's
          top-6 experts and a direct logit attribution, all synchronised layer by layer.
        </p>
      </Prose>

      <h2 className="sec">The components, in the order they act</h2>
      <div className="grid2">
        <Panel title={<><b>01</b> · tokenisation and embedding</>}>
          <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>
            Text is split into tokens, each token is mapped to an integer, and the integer
            indexes an embedding table. A position signal is added (or, in RoPE-based
            attention, injected into the query/key geometry). The result is a
            <code> T × d_model</code> matrix: one vector per position.
          </p>
        </Panel>
        <Panel title={<><b>02</b> · the residual stream</>}>
          <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>
            The stream is the only object every block shares. Each sublayer reads the
            current state, computes a write, and adds it back. This is why a late layer can
            use a feature discovered by an early layer without a separate connection.
          </p>
        </Panel>
        <Panel title={<><b>03</b> · attention: read from the sequence</>}>
          <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>
            A query at the current position scores every allowed key. Softmax turns the
            scores into weights; the head returns a weighted sum of values. The result is
            mixed by an output projection and added to the stream.
          </p>
        </Panel>
        <Panel title={<><b>04</b> · MLP or MoE: transform the token</>}>
          <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>
            A dense MLP applies two learned matrices with a nonlinearity. DeepSeek replaces
            most dense MLPs with a mixture of experts: a router chooses a few experts, and
            their weighted outputs are added to the shared expert.
          </p>
        </Panel>
        <Panel title={<><b>05</b> · normalisation</>}>
          <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>
            RMSNorm keeps the scale of the stream controlled before each sublayer. It does
            not mix positions or features; it rescales the vector so later matrix
            multiplications see a stable magnitude.
          </p>
        </Panel>
        <Panel title={<><b>06</b> · unembedding and softmax</>}>
          <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>
            The final state is projected to vocabulary logits. Softmax turns the logits at
            the last position into a probability distribution. The model does not “choose a
            word” internally; it writes a score for every possible next token.
          </p>
        </Panel>
      </div>

      <Prose>
        <p className="aside">
          <strong>The one-sentence version.</strong> A transformer is a learned sequence of
          residual writes: attention moves information between positions, the MLP transforms
          information within a position, and the unembedding reads the final state as a
          distribution over the next token. Everything else in this atlas is a way of
          looking at one of those writes.
        </p>
      </Prose>

      <Method label="What is measured and what is not">
        <p>No new model run is used on this page. The V2-Lite counts are from the official config and the capture described in Methods; the diagram is a schematic of the forward path.</p>
      </Method>
      <Footnotes items={[
        "Vaswani et al., Attention Is All You Need, arXiv:1706.03762; Elhage et al., A Mathematical Framework for Transformer Circuits, 2021.",
        "The residual-stream framing follows the transformer-circuits literature. It is a useful abstraction, not a claim that the stream is literally a single scalar channel.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={24} />
    </>
  );
}

export function TheoryAttentionPage() {
  return (
    <>
      <ChapterHead n="26" part="V" kicker="Theory · attention"
        title={<>Attention is a <em>weighted read</em></>}
        standfirst={<>A query is a question, a key is an address, and a value is the content. Attention scores the addresses, normalises the scores, and reads a weighted sum of the contents. The softmax is what makes it a comparison rather than a lookup.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>Q · K · V</Chip><Chip tone="teal">softmax</Chip></>}
      />
      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="Query" v="What to look for" s="one vector per position" />
        <Stat k="Key" v="What is available" s="one vector per position" />
        <Stat k="Value" v="What to read" s="weighted sum becomes the output" />
      </div>
      <Prose>
        <p className="lede">
          Attention does not move a token. It computes a new vector for the current
          position by reading from other positions. The read is soft: every allowed
          position receives a weight, and the output is the weighted average of the value
          vectors. A hard lookup would choose one position; attention keeps a distribution.
        </p>
        <p>
          The score is a dot product because it measures alignment: a query and key that
          point in similar directions get a larger score. Dividing by <code>√d</code> keeps
          the variance from growing with head width; the causal mask removes future
          positions before the softmax. The output projection <code>W_O</code> mixes the
          head's read back into the residual stream.
        </p>
      </Prose>
      <Panel title={<><b>Figure 26.1</b> · query, key, value</>}
        note="The attention weights are a probability distribution over the allowed positions. They are not the model's explanation, and the value vectors are not the tokens themselves.">
        <AttentionDiagram />
      </Panel>
      <Panel title={<><b>Figure 26.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="Attention · query to residual write" steps={TASKS.attention} />
      </Panel>
      <Prose>
        <p className="aside">
          <strong>The formula is small; the interpretation is not.</strong>{" "}
          <code>softmax(QKᵀ / √d + mask) V</code> describes the arithmetic. It does not tell
          you what a head means. A head can attend to the previous token and still carry a
          syntactic feature, or attend to a name and write something unrelated to that name.
          The measured attention pages are there to keep the diagram honest.
        </p>
      </Prose>
      <Footnotes items={[
        "The QK/OV decomposition is from Elhage et al., A Mathematical Framework for Transformer Circuits. QK selects where to read; OV selects what to write.",
        "The causal mask is a lower-triangular mask: position t can read positions ≤ t, including itself.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={26} />
    </>
  );
}

export function TheorySelfAttentionPage() {
  const [selected, setSelected] = useState(4);
  const tokens = ["The", "capital", "of", "France", "is"];
  return (
    <>
      <ChapterHead n="27" part="V" kicker="Theory · self-attention"
        title={<>A sequence reading <em>itself</em></>}
        standfirst={<>Self-attention means the queries, keys and values all come from the same sequence. The causal mask is the crucial asymmetry: a token may read the past and itself, never the future. That is what makes a language model trainable one token at a time.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>causal mask</Chip><Chip tone="teal">parallel over positions</Chip></>}
      />
      <div className="row" style={{ gap: 10, marginBottom: 12 }}>
        <span className="mini-label">query position</span>
        <select className="ctl" value={selected} onChange={(e) => setSelected(Number(e.target.value))} style={{ maxWidth: 200 }}>
          {tokens.map((t, i) => <option key={t} value={i}>{i} · {t}</option>)}
        </select>
        <span className="caption">Change the row to see which positions are visible.</span>
      </div>
      <Panel title={<><b>Figure 27.1</b> · the causal triangle</>}
        note="Each row is one query position; each column is a key/value position. Filled cells are visible; the empty upper triangle is the future. The highlighted row is the query selected above.">
        <SelfAttentionMask selected={selected} />
      </Panel>
      <Panel title={<><b>Figure 27.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="Self-attention · parallel reads, causal mask" steps={TASKS.self} accent="var(--teal)" />
      </Panel>
      <Prose>
        <p>
          “Self” is not a special kind of attention; it is attention where the source and
          target sequence are the same. Encoder–decoder attention is the other case: a
          decoder query reads encoder keys and values. DeepSeek's decoder blocks are
          self-attention blocks with a causal mask.
        </p>
        <p className="aside">
          <strong>Why the mask matters.</strong> Without it, the model could read the token
          it is trying to predict. The mask is removed only at inference for the current
          position; the model never gets to look ahead. This is also why the attention
          matrix is triangular during training.
        </p>
      </Prose>
      <Footnotes items={[
        "Self-attention is parallel over the sequence during training; the causal mask simulates the prefix constraint without a recurrence.",
        "The highlighted row is illustrative. The measured attention maps for V2-Lite are in chapter 11.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={27} />
    </>
  );
}

export function TheoryMultiHeadPage() {
  return (
    <>
      <ChapterHead n="28" part="V" kicker="Theory · multi-head attention"
        title={<>Many heads, <em>one residual</em></>}
        standfirst={<>A single attention head has one QK pattern and one OV write. Multiple heads let the model run several reads in parallel in different subspaces, then concatenate the results and mix them with an output projection.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>16 heads in V2-Lite</Chip><Chip tone="teal">concat → W_O</Chip></>}
      />
      <Prose>
        <p className="lede">
          Heads are not separate models. They share the residual stream and the same input
          state, but each has its own learned QK and OV maps. One head may track the
          previous token, another may copy a name, another may look at a delimiter. The
          concatenation is a bookkeeping step; <code>W_O</code> is where the head outputs
          become a single write.
        </p>
      </Prose>
      <Panel title={<><b>Figure 28.1</b> · parallel reads, one write</>}
        note="Four illustrative heads. The real V2-Lite block has 16 query heads and a shared 512-dimensional KV latent; the diagram is about the parallel/concatenate pattern, not the exact tensor shape.">
        <MultiHeadDiagram />
      </Panel>
      <Panel title={<><b>Figure 28.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="Multi-head · parallel subspaces, one write" steps={TASKS.multi} accent="var(--plum)" />
      </Panel>
      <div className="grid3" style={{ marginBottom: 14 }}>
        <Stat k="Why multiple heads" v="Different relations" s="parallel subspaces" />
        <Stat k="What is shared" v="Residual stream" s="not the head weights" />
        <Stat k="What mixes" v="W_O" s="concat → residual" />
      </div>
      <Prose>
        <p className="aside">
          <strong>“Head” is an accounting unit, not a guaranteed concept.</strong> A head can
          be polysemantic, and two heads can implement the same computation redundantly.
          The measured attention chapter uses head statistics — sinks, previous-token
          scores, induction — as evidence, not as a fixed taxonomy.
        </p>
      </Prose>
      <Footnotes items={[
        "The term “head” refers to one attention map and its value projection. Multi-head attention concatenates the head outputs before W_O.",
        "V2-Lite's MLA shares the KV latent across heads; the query heads remain distinct. See chapter 12.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={28} />
    </>
  );
}

export function TheoryMoePage() {
  return (
    <>
      <ChapterHead n="29" part="V" kicker="Theory · mixture of experts"
        title={<>A large memory, <em>sparsely read</em></>}
        standfirst={<>A mixture of experts replaces one feed-forward network with many. A router chooses a small top-k set for each token, so the model can store far more parameters than it pays for on any one forward pass.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>top-k routing</Chip><Chip tone="teal">active ≠ stored</Chip></>}
      />
      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="Stored" v="All experts" s="the capacity" />
        <Stat k="Active" v="top-k + shared" s="the compute per token" />
        <Stat k="Decision" v="Router" s="input-dependent, discrete" />
      </div>
      <Prose>
        <p className="lede">
          In a dense transformer every token pays for the same MLP. In a sparse MoE, the
          router computes a score for every expert, keeps the top <code>k</code>, and adds
          their weighted outputs. The unused experts are still stored in memory; they are
          not computed for this token. This is why a model can have hundreds of billions of
          stored parameters and a much smaller active count.
        </p>
        <p>
          DeepSeek adds a <strong>shared expert</strong> that runs for every token. The
          routed experts provide specialised capacity; the shared expert carries common
          computation that would otherwise be duplicated or lost. The router is part of
          the mechanism, not an implementation detail: a circuit that names “the MLP at
          layer 8” is incomplete until it says which experts fired.
        </p>
      </Prose>
      <Panel title={<><b>Figure 29.1</b> · router, top-k and shared expert</>}
        note="The grid shows stored capacity, not measured routing. V2-Lite has 64 routed experts, top-6, and 2 shared experts; V3 has 256 routed, top-8, 1 shared. The measured routing maps are in chapter 9.">
        <MoeDiagram />
      </Panel>
      <Panel title={<><b>Figure 29.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="MoE · route, compute, combine" steps={TASKS.moe} accent="var(--ember)" />
      </Panel>
      <Prose>
        <p className="aside">
          <strong>The sparsity trap.</strong> Stored parameters and active parameters answer
          different questions. Stored capacity determines what can be learned and how much
          memory inference needs; active compute determines FLOPs and latency. Quoting one
          without the other turns an architectural trade-off into a marketing number.
        </p>
      </Prose>
      <Footnotes items={[
        "DeepSeekMoE uses fine-grained expert segmentation and shared-expert isolation; see the V2 and V3 technical reports.",
        "Routing is input-dependent and discrete. The top-k weights are not a topic taxonomy; see the measured routing chapter for the limits.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={29} />
    </>
  );
}

export function TheorySparsePage() {
  return (
    <>
      <ChapterHead n="30" part="V" kicker="Theory · sparse attention"
        title={<>Reading <em>less of the context</em></>}
        standfirst={<>Full attention compares every query with every key. Sparse attention keeps the causal contract but refuses to pay for the whole matrix: it selects a small key set, pools keys, or slides a local window.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>DSA · CSA · HCA</Chip><Chip tone="teal">1M context</Chip></>}
      />
      <Prose>
        <p className="lede">
          The cost of attention has two parts: the score matrix and the cache. DSA (DeepSeek
          Sparse Attention) uses a lightning indexer to choose a small top-k set of keys for
          each query. CSA and HCA compress the keys along the sequence, so one key stands
          for a block of tokens. A sliding window keeps the most recent tokens dense.
        </p>
        <p>
          These are different trade-offs. DSA keeps token resolution but reads fewer keys;
          CSA/HCA keep a broad view but at a coarser resolution; a sliding window keeps
          exact local resolution but forgets the distant past unless another mechanism
          carries it. Calling all three “sparse attention” hides which quantity is being
          saved.
        </p>
      </Prose>
      <Panel title={<><b>Figure 30.1</b> · three ways to avoid a full triangle</>}
        note="Conceptual matrix patterns. DSA's top-k is data-dependent; CSA pools four tokens per key and HCA pools 128; a sliding window is exact but local. The measured long-context arithmetic is in chapter 5.">
        <SparseDiagram />
      </Panel>
      <Panel title={<><b>Figure 30.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="Sparse attention · choose what to read" steps={TASKS.sparse} accent="var(--signal)" />
      </Panel>
      <Prose>
        <p className="aside">
          <strong>What “sparse” does not mean.</strong> It does not mean the model attends
          to fewer concepts, and it does not mean the missing keys are irrelevant. It means
          the computation has a budget and a selection rule. The selection rule is part of
          the architecture and belongs in the claim.
        </p>
      </Prose>
      <Footnotes items={[
        "DSA was introduced with DeepSeek-V3.2-Exp; CSA/HCA and CED appear in the V4 family material. The measured 1M arithmetic is config-derived.",
        "Sparse attention changes both FLOPs and the KV working set; the cache is a separate budget from the score matrix.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={30} />
    </>
  );
}

export function TheoryLatentPage() {
  return (
    <>
      <ChapterHead n="31" part="V" kicker="Theory · compressed latent"
        title={<>The cache as a <em>latent</em></>}
        standfirst={<>A conventional KV cache stores a key and a value for every head. MLA stores one compressed latent per token and reconstructs the per-head keys and values when needed. The saving is in what must persist between tokens.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>MLA</Chip><Chip tone="teal">512 + 64 channels</Chip></>}
      />
      <Prose>
        <p className="lede">
          Autoregressive inference keeps past keys and values so the model does not recompute
          them. That cache grows with context, layers, heads and head width. Multi-head latent
          attention compresses the per-token state into a low-rank latent plus a small
          positional component, then projects it back into per-head keys and values inside
          each attention call.
        </p>
        <p>
          The compression is not free: the model must learn a bottleneck that preserves what
          the heads need. But the cache arithmetic changes by a large factor. V2-Lite stores
          512 latent channels plus 64 RoPE channels per token, instead of two 16-head
          key/value tensors.
        </p>
      </Prose>
      <Panel title={<><b>Figure 31.1</b> · what persists between tokens</>}
        note="Schematic. The latent is shared across query heads; the per-head K/V tensors are reconstructed on the fly. The measured latent geometry is in chapter 12 and the cache arithmetic in chapter 4.">
        <LatentDiagram />
      </Panel>
      <Panel title={<><b>Figure 31.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="MLA · compress, store, reconstruct" steps={TASKS.latent} accent="var(--teal)" />
      </Panel>
      <Prose>
        <p className="aside">
          <strong>The right comparison is bytes per token, not “compression ratio”.</strong>{" "}
          A ratio depends on the baseline (MHA, GQA, fp16, fp4, sliding window). The atlas
          reports the baseline with the number because a cache claim without a precision
          and a window is not comparable.
        </p>
      </Prose>
      <Footnotes items={[
        "MLA is described in the DeepSeek-V2 technical report. The V2-Lite config gives q_lora_rank 0 and kv_lora_rank 512; the measured cache is 30.4 KiB/token at fp16.",
        "A latent is not automatically interpretable. The measured latent page reports effective rank and variance, not a semantic codebook.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={31} />
    </>
  );
}

export function TheoryCedPage() {
  return (
    <>
      <ChapterHead n="32" part="V" kicker="Theory · causal encoder–decoder"
        title={<>An encoder for the <em>context</em></>}
        standfirst={<>CED splits the stack into a causal encoder that reads the prefix and a decoder that generates the continuation. The encoder can compute a full cache once; the decoder can work from a much smaller shared state.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>V4.1</Chip><Chip tone="teal">20 + 20 layers</Chip></>}
      />
      <Prose>
        <p className="lede">
          “Encoder–decoder” usually means two different sequences: an encoder reads a source
          sentence, a decoder generates a target sentence. CED is a causal variant inside a
          language model: the encoder reads the input prefix, and the decoder continues it.
          The division lets the expensive full-context cache be computed only in a few
          layers while the rest of the stack uses a compressed shared cache.
        </p>
        <p>
          V4.1's 40 layers split into 20 encoder and 20 decoder layers. Only four of the 40
          compute a full input cache (three encoder, one decoder), and every layer keeps a
          128-token sliding window. The number is a published architecture claim, not a
          V2-Lite measurement.
        </p>
      </Prose>
      <Panel title={<><b>Figure 32.1</b> · encode once, decode locally</>}
        note="Schematic of the published V4.1 split. The arrows show the shared cache and the local window, not tensor shapes. The exact layer assignment is in the V4.1 material.">
        <CedDiagram />
      </Panel>
      <Panel title={<><b>Figure 32.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="CED · encode, share, decode" steps={TASKS.ced} accent="var(--signal-deep)" />
      </Panel>
      <Prose>
        <p className="aside">
          <strong>CED is not just sparse attention.</strong> Sparse attention changes which
          keys a query reads. CED changes which layers compute and share a cache. Both reduce
          cost, but they operate on different objects: a read pattern versus a state that
          must persist.
        </p>
      </Prose>
      <Footnotes items={[
        "The CED description follows the V4.1 release material cited in Methods (arXiv:2609.19969): 40 layers as 20 causal encoder + 20 decoder, 4/40 full input cache, 128-token sliding window.",
        "No V4.1 forward pass exists in this atlas. The page explains the published architecture; it does not measure its internals.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={32} />
    </>
  );
}

export function TheoryReasoningPage() {
  return (
    <>
      <ChapterHead n="33" part="V" kicker="Theory · reasoning post-training"
        title={<>Reasoning is a <em>post-train</em></>}
        standfirst={<>R1 is not a new transformer block. It starts from the V3 base and changes the training distribution: long reasoning traces, verifiable rewards and reinforcement learning. The architecture is the control; the post-training is the intervention.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>V3 → R1</Chip><Chip tone="teal">same block shape</Chip></>}
      />
      <Prose>
        <p className="lede">
          A base language model predicts the next token. Reasoning post-training changes
          what it is rewarded for producing: intermediate steps, self-checking and long
          traces that arrive at an answer. If the block shape is fixed, a capability
          difference between V3 and R1 is a training explanation before it is an
          architecture explanation.
        </p>
        <p>
          This is why R1 belongs in the Synthesis as a control branch, not as a fourth
          architectural substitution. It keeps 61 layers, 256 routed experts, top-8 routing
          and one shared expert. The new object is the data and the objective, not the
          attention or the router.
        </p>
      </Prose>
      <Panel title={<><b>Figure 33.1</b> · the same block, a different training loop</>}
        note="Conceptual. R1's architecture is shared with V3; no R1 internal capture exists in this atlas. The reasoning traces and reward signal are a training story, not a measured routing result.">
        <ReasoningDiagram />
      </Panel>
      <Panel title={<><b>Figure 33.2</b> · animated task</>} note={ANIMATION_NOTE}>
        <TaskAnimation title="Reasoning post-training · sample, verify, update" steps={TASKS.reasoning} accent="var(--ember)" />
      </Panel>
      <Prose>
        <p className="aside">
          <strong>The control is the point.</strong> If two checkpoints share a block, a
          behavioural difference cannot be explained by depth, width or expert count. It can
          still be explained by data, optimisation, post-training or decoding. R1 gives the
          atlas a clean way to say which axis is changing.
        </p>
      </Prose>
      <Footnotes items={[
        "R1's public model card and config share V3's 61-layer, 256-routed-expert, top-8 architecture. R1 is a reasoning post-train of the V3 base.",
        "This atlas did not run R1. The claim is about architecture and training, not about R1's internal mechanism.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={33} />
    </>
  );
}

const TERMS = [
  { t: "Token", d: "The unit the model reads. A token is not a word; it can be a word, a word piece, a byte or a punctuation mark.", tag: "input" },
  { t: "Embedding", d: "A learned vector for each token id. The embedding table is the first lookup in the forward pass.", tag: "input" },
  { t: "Residual stream", d: "The shared vector bus. Every attention and MLP sublayer reads it and adds a write back.", tag: "block" },
  { t: "RMSNorm", d: "A scale-only normalisation. It keeps activations stable without subtracting a mean.", tag: "block" },
  { t: "Logit", d: "An unnormalised score for a vocabulary item. Softmax turns a vector of logits into probabilities.", tag: "output" },
  { t: "Softmax", d: "A normalisation that turns scores into a positive distribution summing to one.", tag: "output" },
  { t: "RoPE", d: "Rotary position embedding: position is injected by rotating query and key vectors.", tag: "position" },
  { t: "KV cache", d: "Stored keys and values from earlier tokens so generation does not recompute the prefix.", tag: "inference" },
  { t: "MLA", d: "Multi-head latent attention: a compressed KV latent shared by the query heads.", tag: "attention" },
  { t: "Router", d: "The gate that scores experts and chooses the top-k for each token.", tag: "MoE" },
  { t: "Top-k", d: "The k highest-scoring experts that actually run for a token.", tag: "MoE" },
  { t: "Shared expert", d: "An expert that runs for every token, alongside the routed experts.", tag: "MoE" },
  { t: "Active parameters", d: "The parameters used for one token's forward pass, not the total stored in the checkpoint.", tag: "economics" },
  { t: "MTP", d: "Multi-token prediction: extra heads trained to predict more than the immediate next token.", tag: "training" },
  { t: "Hash layer", d: "A layer whose expert assignment uses a hash rather than a learned router.", tag: "MoE" },
  { t: "DSA", d: "DeepSeek Sparse Attention: an indexer selects a small top-k set of keys.", tag: "long context" },
  { t: "CSA / HCA", d: "Compressed sparse attention variants that pool keys at 4:1 or 128:1 before reading.", tag: "long context" },
  { t: "CED", d: "Causal encoder–decoder: a split stack that computes a full cache only where needed.", tag: "long context" },
  { t: "Quantisation", d: "Storing weights in fewer bits (NF4, FP8) to fit a model on smaller hardware.", tag: "inference" },
  { t: "Context window", d: "The maximum number of tokens the model can condition on in one call.", tag: "inference" },
];

export function KeywordsPage() {
  return (
    <>
      <ChapterHead n="34" part="V" kicker="Theory · keyword atlas"
        title={<>The vocabulary of <em>the atlas</em></>}
        standfirst={<>A field guide to the words that recur across the report. Each card is a definition you can carry into a figure; the linked chapters are where the term becomes measurable.</>}
        chips={<><Chip tone="blue">visual primer</Chip><Chip>{TERMS.length} terms</Chip><Chip tone="teal">one glance each</Chip></>}
      />
      <Panel title={<><b>Figure 34.1</b> · one block, five vocabularies</>}
        note="The center is the object; the surrounding boxes are the five families of words the atlas uses to describe it. The map is a reading aid, not a measured circuit.">
        <KeywordMap />
      </Panel>
      <Panel title={<><b>Figure 34.2</b> · from config to claim</>}
        note="The evidence ladder makes the difference between a config fact, a forward-pass observation, an attribution and a causal intervention explicit.">
        <KeywordPipeline />
      </Panel>
      <div className="cards">
        {TERMS.map((x) => (
          <Panel key={x.t} title={<><b>{x.t}</b> · {x.tag}</>}>
            <p style={{ fontSize: 12.5, lineHeight: 1.65, color: "var(--ink-2)" }}>{x.d}</p>
          </Panel>
        ))}
      </div>
      <Prose>
        <p className="aside">
          A keyword is not an explanation. “The router chose expert 37” is a fact about a
          gate; “expert 37 means chemistry” is a claim that needs an intervention. The atlas
          uses the first sentence freely and the second only when the measurement supports it.
        </p>
      </Prose>
      <Footnotes items={[
        "Definitions are deliberately operational: they say what the object does in a forward pass, not what it semantically means.",
        SOURCE_NOTE,
      ]} />
      <TheoryFoot index={34} />
    </>
  );
}
