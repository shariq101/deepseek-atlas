import React from "react";

export function Box({
  x, y, w = 140, h = 34, title, sub, tone = "",
}: { x: number; y: number; w?: number; h?: number; title: string; sub?: string; tone?: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} className={"blockbox " + tone} rx={1} />
      <text x={x + 8} y={y + 14} className="blocklabel">{title}</text>
      {sub && <text x={x + 8} y={y + 25} className="blocksub">{sub}</text>}
    </g>
  );
}

export function Arrow({ x1, y1, x2, y2, tone = "var(--ink-3)" }: { x1: number; y1: number; x2: number; y2: number; tone?: string }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={tone} strokeWidth={1.1} />;
}

export function TokenStrip({ y, tokens, x = 28, gap = 54, highlight = -1 }: { y: number; tokens: string[]; x?: number; gap?: number; highlight?: number }) {
  return (
    <g>
      {tokens.map((t, i) => (
        <g key={`${t}-${i}`}>
          <circle cx={x + i * gap} cy={y} r={highlight === i ? 6 : 4} fill={highlight === i ? "var(--signal)" : "var(--paper-2)"} stroke={highlight === i ? "var(--signal)" : "var(--ink-3)"} />
          <text x={x + i * gap} y={y - 9} textAnchor="middle" className="ticklabel">{t}</text>
        </g>
      ))}
    </g>
  );
}

export function TransformerDiagram() {
  const W = 700, H = 570;
  const blocks = [366, 302, 238];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="The transformer stack from tokens to probabilities">
      <text x={26} y={18} className="mini-label">ONE FORWARD PASS · LEFT TO RIGHT, BOTTOM TO TOP</text>
      <TokenStrip y={514} tokens={["The", "capital", "of", "France", "is"]} x={56} gap={62} highlight={4} />
      <text x={26} y={552} className="blocksub">tokens → integer ids → embedding vectors</text>
      <Arrow x1={56} y1={504} x2={56} y2={478} />
      <Box x={20} y={444} w={150} title="embedding + position" sub="T × d_model" tone="tint" />
      <line x1={170} y1={461} x2={180} y2={461} stroke="var(--signal)" strokeWidth={1.4} />

      {/* residual stream runs upward, with one stub into each representative block */}
      <line x1={180} y1={461} x2={180} y2={260} stroke="var(--signal)" strokeWidth={2} />
      <text x={24} y={250} className="mono" style={{ fontSize: 9, fill: "var(--signal)" }}>residual stream</text>

      {blocks.map((y, i) => (
        <g key={i}>
          <line x1={180} y1={y + 22} x2={200} y2={y + 22} stroke="var(--signal)" strokeWidth={1.2} />
          <Box x={200} y={y} w={150} h={44} title={`block ${i + 1}`} sub="attention → MLP" />
          <Box x={370} y={y + 4} w={150} h={36} title="attention" sub="reads the sequence" tone="tint" />
          <Box x={540} y={y + 4} w={130} h={36} title="MLP / MoE" sub="writes features" tone="ember" />
          <Arrow x1={350} y1={y + 22} x2={370} y2={y + 22} />
          <Arrow x1={520} y1={y + 22} x2={540} y2={y + 22} />
        </g>
      ))}
      <Arrow x1={285} y1={366} x2={285} y2={346} />
      <Arrow x1={285} y1={302} x2={285} y2={282} />
      <Arrow x1={285} y1={238} x2={285} y2={214} />

      <Box x={190} y={180} w={190} h={34} title="repeat × N blocks" sub="N = 27 in V2-Lite" tone="tint" />
      <Arrow x1={285} y1={180} x2={285} y2={166} />
      <Box x={190} y={132} w={190} h={34} title="final RMSNorm" sub="one vector per position" />
      <Arrow x1={285} y1={132} x2={285} y2={118} />
      <Box x={190} y={84} w={190} h={34} title="unembedding W_U" sub="d_model → vocab" tone="ember" />
      <Arrow x1={285} y1={84} x2={285} y2={66} />
      <Box x={190} y={36} w={190} h={30} title="logits → softmax" sub="P(next token | context)" tone="ember" />

      <text x={490} y={70} className="blocksub">The residual is the shared bus:</text>
      <text x={490} y={84} className="blocksub">each sublayer reads it and adds its write.</text>
      <text x={490} y={104} className="blocksub">No recurrence; depth is the loop.</text>
      <text x={490} y={124} className="blocksub">Causal mask: position t sees ≤ t.</text>
    </svg>
  );
}

export function AttentionDiagram() {
  const W = 700, H = 410;
  const row = [0.46, 0.08, 0.12, 0.27, 0.07];
  const tokens = ["The", "capital", "of", "France", "is"];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Queries, keys, values and the attention weighted read">
      <defs>
        <marker id="att-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--ink-3)" />
        </marker>
      </defs>
      <text x={20} y={16} className="mini-label">A QUERY SCORES EVERY KEY, THEN READS A WEIGHTED SUM OF VALUES</text>
      <TokenStrip y={54} tokens={tokens} x={44} gap={62} highlight={4} />
      <Arrow x1={110} y1={64} x2={110} y2={92} />
      <Box x={44} y={92} w={132} title="query q_t" sub="what am I looking for?" tone="tint" />
      <Box x={196} y={92} w={150} title="keys k_1 … k_T" sub="what is available?" />
      <Box x={366} y={92} w={150} title="values v_1 … v_T" sub="what would I read?" tone="ember" />
      <Arrow x1={176} y1={108} x2={196} y2={108} />
      <text x={188} y={84} textAnchor="middle" className="mono" style={{ fontSize: 9 }}>·</text>
      <Arrow x1={110} y1={126} x2={196} y2={148} />
      <Box x={44} y={148} w={302} h={30} title="logits = qₜ · kᵢ / √d" sub="one score per earlier token" tone="tint" />
      <Arrow x1={195} y1={178} x2={195} y2={202} />
      <Box x={44} y={202} w={302} h={30} title="softmax over i" sub="weights sum to 1 · causal mask" tone="ember" />
      {row.map((p, i) => (
        <g key={i}>
          <rect x={44 + i * 59} y={248} width={54} height={18 + p * 56} fill="var(--signal)" opacity={0.24 + p * 0.7} />
          <text x={71 + i * 59} y={306} textAnchor="middle" className="ticklabel">{p.toFixed(2)}</text>
          <text x={71 + i * 59} y={320} textAnchor="middle" className="ticklabel">{tokens[i]}</text>
        </g>
      ))}
      <text x={44} y={240} className="mini-label">WEIGHTS</text>
      <Arrow x1={346} y1={217} x2={390} y2={248} />
      <Box x={390} y={242} w={190} h={38} title="Σ aᵢ · vᵢ" sub="the head's read" tone="tint" />
      <Arrow x1={485} y1={280} x2={485} y2={306} />
      <Box x={390} y={306} w={190} h={36} title="output projection W_O" sub="back into residual" tone="ember" />
      <text x={20} y={350} className="blocksub">The attention matrix is not the answer.</text>
      <text x={20} y={364} className="blocksub">It is a routing distribution over values.</text>
      <text x={20} y={378} className="blocksub">A head can attend to a token</text>
      <text x={20} y={392} className="blocksub">without writing anything useful.</text>
    </svg>
  );
}

export function SelfAttentionMask({ selected = 4 }: { selected?: number }) {
  const W = 670, H = 330;
  const tokens = ["The", "capital", "of", "France", "is"];
  const n = tokens.length;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Causal self-attention mask and one selected query row">
      <text x={20} y={16} className="mini-label">SELF-ATTENTION · QUERY ROW FOR “{tokens[selected].toUpperCase()}”</text>
      <TokenStrip y={48} tokens={tokens} x={50} gap={60} highlight={selected} />
      <g transform="translate(54 82)">
        {Array.from({ length: n }, (_, r) =>
          Array.from({ length: n }, (_, c) => {
            const open = c <= r;
            const rowOn = r === selected;
            const weight = open ? (rowOn ? (c === selected ? 0.5 : c === 0 ? 0.24 : 0.1) : 0.16) : 0;
            return (
              <g key={`${r}-${c}`}>
                <rect x={c * 48} y={r * 40} width={43} height={35}
                  fill={open ? `rgba(43,70,240,${0.06 + weight * 1.5})` : "var(--paper-3)"}
                  stroke={rowOn ? "var(--signal)" : "var(--rule)"} strokeWidth={rowOn ? 1.5 : 0.7} />
                {r === 0 && <text x={c * 48 + 21} y={-8} textAnchor="middle" className="ticklabel">{tokens[c]}</text>}
                {c === 0 && <text x={-8} y={r * 40 + 22} textAnchor="end" className="ticklabel">{tokens[r]}</text>}
                {open && <circle cx={c * 48 + 21} cy={r * 40 + 17} r={2} fill={rowOn ? "var(--signal)" : "var(--ink-3)"} opacity={rowOn ? 1 : 0.35} />}
              </g>
            );
          }),
        )}
      </g>
      <text x={318} y={100} className="blocksub">the triangular hole is the causal mask:</text>
      <text x={318} y={114} className="blocksub">a query cannot read the future.</text>
      <text x={318} y={142} className="blocksub">Every token still contributes a query,</text>
      <text x={318} y={156} className="blocksub">a key and a value. “Self” means</text>
      <text x={318} y={170} className="blocksub">they come from the same sequence.</text>
      <text x={318} y={198} className="blocksub">This is parallel over positions;</text>
      <text x={318} y={212} className="blocksub">the mask is what makes it causal.</text>
    </svg>
  );
}

export function MultiHeadDiagram() {
  const W = 700, H = 340;
  const heads = [
    { label: "H0", color: "#2b46f0", pattern: [0.6, 0.1, 0.1, 0.1, 0.1] },
    { label: "H1", color: "#17656b", pattern: [0.1, 0.3, 0.3, 0.1, 0.1] },
    { label: "H2", color: "#6a3f73", pattern: [0.1, 0.1, 0.2, 0.5, 0.1] },
    { label: "H3", color: "#b8461c", pattern: [0.1, 0.1, 0.1, 0.2, 0.5] },
  ];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Multiple attention heads produce parallel reads that are concatenated and projected">
      <text x={20} y={16} className="mini-label">PARALLEL SUBSPACES · EACH HEAD HAS ITS OWN QK AND OV</text>
      <TokenStrip y={48} tokens={["The", "capital", "of", "France", "is"]} x={38} gap={54} highlight={4} />
      {heads.map((h, i) => (
        <g key={h.label}>
          <Box x={20} y={92 + i * 52} w={170} h={38} title={`${h.label} · QK / OV`} sub={i === 0 ? "syntax / previous token" : i === 1 ? "entity tracking" : i === 2 ? "long-range context" : "copy / induction"} tone="tint" />
          {h.pattern.map((p, j) => (
            <rect key={j} x={210 + j * 25} y={98 + i * 52} width={20} height={24} fill={h.color} opacity={0.12 + p * 0.85} />
          ))}
          <text x={342} y={114 + i * 52} className="ticklabel">{h.label} read</text>
        </g>
      ))}
      {[0, 1, 2, 3].map((i) => <line key={`join-${i}`} x1={380} y1={114 + i * 52} x2={390} y2={114 + i * 52} stroke="var(--ink-3)" strokeWidth={1.1} />)}
      <line x1={390} y1={114} x2={390} y2={270} stroke="var(--ink-3)" strokeWidth={1.1} />
      <Arrow x1={390} y1={185} x2={420} y2={185} />
      <Box x={420} y={166} w={150} h={38} title="concat heads" sub="H × d_head" tone="ember" />
      <Arrow x1={495} y1={204} x2={495} y2={230} />
      <Box x={420} y={230} w={150} h={38} title="W_O" sub="back to d_model" tone="ember" />
      <text x={20} y={318} className="blocksub">V2-Lite: 16 query heads, one 512-d KV latent. Concat is a view; W_O mixes the heads back into the residual stream.</text>
    </svg>
  );
}

export function MoeDiagram() {
  const W = 700, H = 380;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="A mixture of experts router choosing top-k experts plus a shared expert">
      <text x={20} y={16} className="mini-label">A TOKEN DOES NOT VISIT EVERY EXPERT</text>
      <TokenStrip y={48} tokens={["capital"]} x={44} gap={54} highlight={0} />
      <Arrow x1={50} y1={60} x2={50} y2={86} />
      <Box x={20} y={86} w={150} h={36} title="token hidden state" sub="d_model" tone="tint" />
      <Arrow x1={170} y1={104} x2={210} y2={104} />
      <Box x={210} y={86} w={150} h={36} title="router W_g" sub="score every expert" />
      <Arrow x1={285} y1={122} x2={285} y2={148} />
      <Box x={205} y={148} w={160} h={34} title="top-6 of 64" sub="weights + indices" tone="ember" />
      <g transform="translate(390 76)">
        {Array.from({ length: 8 }, (_, r) =>
          Array.from({ length: 8 }, (_, c) => {
            const on = [[0, 0], [1, 2], [2, 5], [3, 1], [4, 6], [5, 3]].some(([rr, cc]) => rr === r && cc === c);
            return <rect key={`${r}-${c}`} x={c * 28} y={r * 28} width={24} height={24} fill={on ? "var(--signal)" : "var(--paper-2)"} stroke="var(--rule)" opacity={on ? 0.9 : 0.8} />;
          }),
        )}
        <text x={0} y={-8} className="mini-label">64 ROUTED EXPERTS</text>
      </g>
      <Arrow x1={390} y1={148} x2={350} y2={190} />
      <Box x={200} y={190} w={180} h={38} title="weighted expert sum" sub="top-6 routed writes" tone="tint" />
      <Box x={200} y={248} w={180} h={38} title="+ shared expert" sub="always on · 2 in V2-Lite" tone="ember" />
      <Arrow x1={290} y1={228} x2={290} y2={248} />
      <Arrow x1={290} y1={286} x2={290} y2={312} />
      <Box x={200} y={312} w={180} h={36} title="+ residual" sub="one block write" tone="ember" />
      <text x={390} y={330} className="blocksub">router: a discrete decision; stored capacity = 64 experts.</text>
      <text x={390} y={344} className="blocksub">paid compute = 6 experts + shared.</text>
      <text x={390} y={358} className="blocksub">that gap is the sparsity dividend.</text>
      <text x={390} y={372} className="blocksub">V3 256 · top-8 · 1 shared. V4.1 384 · top-6 · 1 shared.</text>
    </svg>
  );
}

export function SparseDiagram() {
  const W = 700, H = 270;
  const panels = [
    { x: 20, label: "dense", sub: "T × T", cells: (r: number, c: number) => c <= r, color: "var(--ink-3)" },
    { x: 240, label: "DSA", sub: "top-2048 keys", cells: (r: number, c: number) => c <= r && (c > r - 4 || c % 3 === 0), color: "var(--signal)" },
    { x: 460, label: "CSA / HCA", sub: "pooled keys", cells: (r: number, c: number) => c <= r && c % 3 === 0, color: "var(--ember)" },
  ];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Dense attention compared with sparse index selection and pooled compressed attention">
      <text x={20} y={16} className="mini-label">THE SAME CAUSAL TRIANGLE, THREE DIFFERENT READ PATTERNS</text>
      {panels.map((p) => (
        <g key={p.label} transform={`translate(${p.x} 38)`}>
          <text x={0} y={0} className="blocklabel">{p.label}</text>
          <text x={0} y={12} className="blocksub">{p.sub}</text>
          {Array.from({ length: 12 }, (_, r) =>
            Array.from({ length: 12 }, (_, c) => {
              const on = p.cells(r, c);
              return <rect key={`${r}-${c}`} x={c * 14} y={22 + r * 14} width={12} height={12}
                fill={on ? p.color : "var(--paper-3)"} opacity={on ? 0.8 : 0.45} stroke="var(--paper)" strokeWidth={0.5} />;
            }),
          )}
        </g>
      ))}
      <text x={20} y={238} className="blocksub">DSA selects a small set of keys per query; CSA pools 4 tokens per key; HCA pools 128.</text>
      <text x={20} y={252} className="blocksub">A sliding window keeps the recent past dense.</text>
    </svg>
  );
}

export function LatentDiagram() {
  const W = 700, H = 360;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="MLA compressed latent compared with per-head key and value caches">
      <text x={20} y={16} className="mini-label">MLA · THE CACHE IS A SHARED LATENT, NOT PER-HEAD K/V</text>
      <Box x={20} y={48} w={190} h={44} title="conventional MHA cache" sub="2 × heads × head_dim per token" />
      <g transform="translate(20 108)">
        {Array.from({ length: 6 }, (_, i) => <rect key={i} x={i * 30} y={0} width={25} height={92} fill="var(--rule-2)" opacity={0.55} />)}
        {Array.from({ length: 6 }, (_, i) => <rect key={`v${i}`} x={i * 30} y={98} width={25} height={92} fill="var(--ember)" opacity={0.25} />)}
        <text x={90} y={-8} textAnchor="middle" className="ticklabel">keys</text>
        <text x={90} y={88} textAnchor="middle" className="ticklabel">values</text>
      </g>
      <Box x={270} y={48} w={210} h={44} title="MLA cache" sub="latent 512 + rope 64 per token" tone="tint" />
      <g transform="translate(270 108)">
        {Array.from({ length: 8 }, (_, i) => <rect key={i} x={i * 24} y={0} width={20} height={150} fill="var(--signal)" opacity={0.12 + i * 0.08} />)}
        <text x={96} y={168} textAnchor="middle" className="ticklabel">one latent vector</text>
        <text x={96} y={184} textAnchor="middle" className="ticklabel">read by every head</text>
      </g>
      <Arrow x1={480} y1={150} x2={520} y2={150} />
      <Box x={520} y={112} w={150} h={74} title="W_uk / W_uv" sub="reconstruct per-head K/V" tone="ember" />
      <text x={20} y={334} className="blocksub">The cache stores the compressed representation; the per-head keys and values are reconstructed when needed.</text>
      <text x={20} y={350} className="blocksub">V2-Lite: 512 latent + 64 rope = 576 channels/token; 30.4 KiB at fp16, 7.1× below MHA.</text>
    </svg>
  );
}

export function CedDiagram() {
  const W = 700, H = 310;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Causal encoder-decoder architecture splitting context encoding and token generation">
      <text x={20} y={16} className="mini-label">CED · ENCODE THE CONTEXT ONCE, DECODE WITH A SMALLER WORKING SET</text>
      <Box x={20} y={44} w={260} h={48} title="causal encoder" sub="reads the input prefix · writes a full cache" tone="tint" />
      <Box x={20} y={112} w={260} h={48} title="shared KV blocks" sub="only a few layers compute the global cache" tone="ember" />
      <Box x={380} y={44} w={260} h={48} title="decoder" sub="generates the continuation · reads the shared cache" />
      <Box x={380} y={112} w={260} h={48} title="local sliding window" sub="recent tokens stay dense" />
      <Arrow x1={280} y1={68} x2={380} y2={68} />
      <Arrow x1={280} y1={136} x2={380} y2={136} />
      <Arrow x1={150} y1={92} x2={150} y2={112} />
      <Arrow x1={510} y1={92} x2={510} y2={112} />
      <text x={20} y={202} className="blocksub">V4.1: 40 layers = 20 encoder + 20 decoder. Only 4 of 40 layers (3 encoder, 1 decoder) compute a full input cache.</text>
      <text x={20} y={220} className="blocksub">Every layer still keeps a 128-token sliding window. The global attention cost is paid sparsely.</text>
      <text x={20} y={250} className="mini-label">WHY IT IS NOT JUST “SPARSE ATTENTION”</text>
      <text x={20} y={268} className="blocksub">The encoder and decoder have different jobs. CED is a division of labour, not only a mask on a matrix.</text>
    </svg>
  );
}

export function ReasoningDiagram() {
  const W = 700, H = 310;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="V3 base model to R1 reasoning post-training as a training branch">
      <text x={20} y={16} className="mini-label">R1 IS A TRAINING BRANCH, NOT A NEW BLOCK</text>
      <Box x={30} y={60} w={170} h={54} title="V3 base" sub="671B total / 37B active · 61 layers" tone="tint" />
      <Arrow x1={200} y1={87} x2={280} y2={87} />
      <Box x={280} y={44} w={200} h={86} title="reasoning post-training" sub="long traces · verifiable rewards · RL" tone="ember" />
      <Arrow x1={480} y1={87} x2={560} y2={87} />
      <Box x={560} y={60} w={120} h={54} title="R1" sub="same block shape" tone="tint" />
      <path d="M390 132 C390 158 470 158 470 132" fill="none" stroke="var(--ember)" strokeWidth={1.2} strokeDasharray="3 4" />
      <text x={490} y={150} className="blocksub">rollouts → reward → update</text>
      <text x={490} y={164} className="blocksub">reasoning traces are the new data</text>
      <text x={30} y={230} className="blocksub">What changes: the post-training objective and the distribution of traces.</text>
      <text x={30} y={246} className="blocksub">What does not: 61 layers, 256 routed experts, top-8, one shared expert.</text>
      <text x={30} y={272} className="blocksub">A capability difference is therefore a training explanation before it is an architecture explanation.</text>
    </svg>
  );
}

export function KeywordMap() {
  const W = 700, H = 430;
  const nodes = [
    { x: 34, y: 54, w: 176, title: "Attention", a: "self · multi-head", b: "DSA · CSA / HCA", tone: "tint" },
    { x: 490, y: 54, w: 176, title: "MoE", a: "router · top-k", b: "shared · balance", tone: "ember" },
    { x: 22, y: 250, w: 170, title: "Cache", a: "MLA latent · KV", b: "sliding window", tone: "tint" },
    { x: 508, y: 250, w: 170, title: "Post-training", a: "SFT · RL · GRPO", b: "reasoning traces", tone: "ember" },
    { x: 270, y: 342, w: 160, title: "Evidence", a: "lens · DLA", b: "ablation · ACDC", tone: "tint" },
  ];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="A keyword map connecting attention, mixture of experts, cache, post-training and evidence to the transformer block">
      <text x={350} y={16} textAnchor="middle" className="mini-label">FIVE VOCABULARIES · ONE OBJECT</text>
      <line x1={300} y1={182} x2={210} y2={112} stroke="var(--rule-2)" strokeWidth={1} />
      <line x1={400} y1={182} x2={490} y2={112} stroke="var(--rule-2)" strokeWidth={1} />
      <line x1={300} y1={224} x2={192} y2={270} stroke="var(--rule-2)" strokeWidth={1} />
      <line x1={400} y1={224} x2={508} y2={270} stroke="var(--rule-2)" strokeWidth={1} />
      <line x1={350} y1={262} x2={350} y2={342} stroke="var(--rule-2)" strokeWidth={1} />
      <circle cx={350} cy={200} r={62} fill="var(--paper-2)" stroke="var(--ink)" strokeWidth={1} />
      <text x={350} y={188} textAnchor="middle" className="mini-label">THE BLOCK</text>
      <text x={350} y={204} textAnchor="middle" className="blocklabel">residual stream</text>
      <text x={350} y={218} textAnchor="middle" className="blocksub">the shared bus</text>
      {nodes.map((n) => (
        <g key={n.title}>
          <rect x={n.x} y={n.y} width={n.w} height={70} rx={1} className={"blockbox " + n.tone} />
          <text x={n.x + 10} y={n.y + 20} className="blocklabel">{n.title}</text>
          <text x={n.x + 10} y={n.y + 36} className="blocksub">{n.a}</text>
          <text x={n.x + 10} y={n.y + 49} className="blocksub">{n.b}</text>
        </g>
      ))}
      <text x={350} y={424} textAnchor="middle" className="blocksub">Every card below defines one of these words in terms of a forward pass, not a metaphor.</text>
    </svg>
  );
}

export function VerbFrameDiagram() {
  const W = 700, H = 342;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="The verb put selecting agent, theme and goal roles">
      <text x={20} y={16} className="mini-label">A PREDICATE SELECTS A FRAME OF PARTICIPANTS</text>
      <text x={20} y={44} className="blocksub">“The curator put the painting in the vault.”</text>
      <Box x={260} y={76} w={180} h={54} title="put" sub="predicate · three roles" tone="ember" />
      <Arrow x1={326} y1={130} x2={104} y2={206} tone="var(--ember)" />
      <Arrow x1={350} y1={130} x2={350} y2={206} tone="var(--ember)" />
      <Arrow x1={374} y1={130} x2={586} y2={206} tone="var(--ember)" />
      <Box x={24} y={206} w={160} h={46} title="AGENT" sub="The curator · who acts" tone="tint" />
      <Box x={260} y={206} w={180} h={46} title="THEME" sub="the painting · what moves" tone="tint" />
      <Box x={500} y={206} w={176} h={46} title="GOAL" sub="in the vault · where it ends" tone="tint" />
      <text x={20} y={286} className="blocksub">The preposition “in” introduces the goal, but the verb's frame is what makes a goal obligatory.</text>
      <text x={20} y={302} className="blocksub">sleep: agent only  ·  put: agent, theme, goal  ·  give: agent, theme, recipient</text>
      <text x={20} y={326} className="blocksub">This is a linguistic description. It is not a claim that one head implements one role.</text>
    </svg>
  );
}

export function SarcasmDiagram() {
  const W = 700, H = 310;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Literal sentiment and context combining into a sarcastic intended meaning">
      <text x={20} y={16} className="mini-label">THE CUE IS THE MISMATCH, NOT THE WORD “GREAT”</text>
      <Box x={22} y={46} w={200} h={72} title="literal utterance" sub="“Great. Another all-nighter.”" tone="tint" />
      <Box x={250} y={46} w={200} h={72} title="shared context" sub="deadline moved to Friday" />
      <Box x={478} y={46} w={200} h={72} title="speaker meaning" sub="negative · “this is bad for us”" tone="ember" />
      <Arrow x1={122} y1={118} x2={292} y2={174} tone="var(--ember)" />
      <Arrow x1={350} y1={118} x2={350} y2={174} tone="var(--ember)" />
      <Box x={250} y={174} w={200} h={52} title="incongruity" sub="positive wording × negative consequence" tone="ember" />
      <Arrow x1={450} y1={200} x2={478} y2={120} tone="var(--ember)" />
      <text x={22} y={258} className="blocksub">A model that keys on “great” as positive sentiment inverts the intended meaning.</text>
      <text x={22} y={274} className="blocksub">A model that keys on the deadline may classify correctly without understanding irony.</text>
      <text x={22} y={296} className="blocksub">Benchmark labels, context length and speaker intent all belong in the claim.</text>
    </svg>
  );
}

export function ContextCarryDiagram() {
  const W = 700, H = 386;
  const sentences = [
    { s: "S1", a: "Mira leads", b: "the lab." },
    { s: "S2", a: "She hired", b: "Noor." },
    { s: "S3", a: "Noor runs", b: "the assay." },
    { s: "S4", a: "The review", b: "moved." },
    { s: "S5", a: "Who leads", b: "the lab?" },
  ];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="A later query reconstructing an earlier sentence through the KV cache and attention">
      <text x={20} y={16} className="mini-label">THE PAST IS RECONSTRUCTED FROM CACHED STATES, NOT PASSED AS ONE SUMMARY</text>
      {sentences.map((x, i) => (
        <g key={x.s}>
          <rect x={20 + i * 130} y={34} width={120} height={54} className={"blockbox " + (i === 4 ? "tint" : "")} rx={1} />
          <text x={30 + i * 130} y={48} className="ticklabel">{x.s}</text>
          <text x={30 + i * 130} y={63} className="blocklabel">{x.a}</text>
          <text x={30 + i * 130} y={77} className="blocksub">{x.b}</text>
        </g>
      ))}
      <text x={20} y={112} className="mini-label">KV CACHE · ONE ENTRY PER TOKEN PER LAYER</text>
      {sentences.map((x, i) => (
        <g key={`kv-${x.s}`}>
          <rect x={20 + i * 130} y={124} width={120} height={38} className={"blockbox " + (i === 0 ? "ember" : "")} rx={1} />
          <text x={30 + i * 130} y={139} className="blocksub">{i === 0 ? "Mira → K/V" : i === 4 ? "query → q" : "K/V stored"}</text>
          <text x={30 + i * 130} y={153} className="blocksub">{i === 4 ? "the current read" : `layer 1 … 27`}</text>
        </g>
      ))}
      <path d="M80 162 C80 198 600 198 600 162" fill="none" stroke="var(--signal)" strokeWidth={1.2} strokeDasharray="3 4" />
      <text x={340} y={213} textAnchor="middle" className="blocksub">a query at S5 reads a cached S1 key, then a value</text>
      <Box x={20} y={232} w={200} h={54} title="current residual · S5" sub="27 blocks update only this position" tone="tint" />
      <Arrow x1={220} y1={259} x2={248} y2={259} tone="var(--signal)" />
      <Box x={248} y={232} w={190} h={54} title="attention read" sub="weighted values enter the stream" />
      <Arrow x1={438} y1={259} x2={466} y2={259} tone="var(--signal)" />
      <Box x={466} y={232} w={214} h={54} title="generate · append · repeat" sub="the cache grows with the context" tone="ember" />
      <text x={20} y={318} className="blocksub">What carries: token ids, positions and per-layer key/value state.</text>
      <text x={20} y={334} className="blocksub">What does not: a single paragraph vector or a permanent memory of the conversation.</text>
      <text x={20} y={350} className="blocksub">What can break: distance, interference, eviction, compression and position limits.</text>
    </svg>
  );
}

export function KeywordPipeline() {
  const W = 700, H = 180;
  const nodes = [
    { x: 20, title: "config", a: "shape · count", b: "read the spec", tone: "" },
    { x: 154, title: "capture", a: "hooks · traces", b: "what fires", tone: "tint" },
    { x: 288, title: "attribute", a: "lens · DLA", b: "who wrote it", tone: "tint" },
    { x: 422, title: "intervene", a: "zero · patch", b: "does it matter", tone: "ember" },
    { x: 556, title: "claim", a: "scope · caveat", b: "what survives", tone: "ember" },
  ];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Evidence pipeline from configuration through capture, attribution and intervention to a scoped claim">
      <text x={20} y={18} className="mini-label">FROM CONFIG TO CLAIM · FOUR LEVELS OF EVIDENCE</text>
      {nodes.map((n, i) => (
        <g key={n.title}>
          <rect x={n.x} y={44} width={124} height={66} rx={1} className={"blockbox " + n.tone} />
          <text x={n.x + 8} y={64} className="blocklabel">{n.title}</text>
          <text x={n.x + 8} y={81} className="blocksub">{n.a}</text>
          <text x={n.x + 8} y={95} className="blocksub">{n.b}</text>
          {i < nodes.length - 1 && <Arrow x1={n.x + 124} y1={77} x2={nodes[i + 1].x} y2={77} tone="var(--ink-3)" />}
        </g>
      ))}
      <text x={20} y={140} className="blocksub">Config is not behaviour. Attribution is not causation. An intervention is not a semantic label.</text>
      <text x={20} y={156} className="blocksub">The strongest sentence a page can support is the one whose evidence box it can name.</text>
    </svg>
  );
}
