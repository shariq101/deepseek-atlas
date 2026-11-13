import React, { useState } from "react";
import {
  Architecture as ArchData, ArchFocus, Budget, LayerPlan, compact, fmt, useJson, useMeasure, useOne,
} from "../lib/atlas";
import {
  Caption, ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Seg, Stat,
} from "../components/ui";

const MODELS = [
  { v: "v2lite", label: "V2-Lite" },
  { v: "v3", label: "V3" },
  { v: "v4flash", label: "V4-Flash" },
  { v: "v41flash", label: "V4.1-Flash" },
] as const;

type ModelKey = (typeof MODELS)[number]["v"];

function Box({ x, y, w = 150, h = 34, label, sub, tone = "" }: {
  x: number; y: number; w?: number; h?: number; label: string; sub?: string; tone?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} className={"blockbox " + tone} />
      <text x={x + 8} y={y + 14} className="blocklabel">{label}</text>
      {sub && <text x={x + 8} y={y + 25} className="blocksub">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} className="flow" />;
}

function MlaDiagram({ f }: { f: ArchFocus }) {
  const W = 520, H = 520;
  const qk = f.qkNope + f.qkRope;
  const kvOut = f.kvLora ?? f.headDim;
  return (
    <svg className="arch-svg" viewBox={`0 0 ${W} ${H}`} role="img">
      <defs>
        <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
        </marker>
      </defs>
      <text x={0} y={10} className="mini-label" style={{ fontSize: 8, letterSpacing: "0.1em" }}>SUBBLOCK 1 · MULTI-HEAD LATENT ATTENTION</text>
      <Box x={0} y={22} label="residual x" sub={`T × ${f.hidden}`} w={135} tone="" />
      <Arrow x1={135} y1={39} x2={180} y2={39} />
      <Box x={180} y={22} label="RMSNorm" sub={`${f.hidden} gain`} w={105} />
      <Arrow x1={285} y1={39} x2={320} y2={39} />
      <Box x={320} y={22} label="W_dq" sub={f.qLora ? `${f.hidden} × ${f.qLora}` : "—"} w={105} tone={f.qLora ? "" : "tint"} />
      {f.qLora && (
        <>
          <Arrow x1={372} y1={56} x2={372} y2={78} />
          <Box x={300} y={78} label="RMSNorm + W_uq" sub={`${f.qLora} → ${f.heads}×${qk}`} w={145} tone="tint" />
        </>
      )}
      <Arrow x1={372} y1={112} x2={372} y2={132} />
      <Box x={265} y={132} label="Q" sub={`${f.heads} heads × ${qk}`} w={215} tone="tint" />

      <Arrow x1={180} y1={56} x2={180} y2={72} />
      <Box x={0} y={72} label="W_dkv" sub={`${f.hidden} × ${kvOut + f.qkRope}`} w={165} />
      <Arrow x1={82} y1={106} x2={82} y2={122} />
      <Box x={0} y={122} label="split → latent" sub={`T × ${kvOut}`} w={120} />
      <Box x={130} y={122} label="k_pe" sub={`T × ${f.qkRope}`} w={90} />
      <Arrow x1={60} y1={156} x2={60} y2={174} />
      <Box x={0} y={174} label="RMSNorm + W_ukv" sub={`${kvOut} → ${f.heads}×(${f.qkNope}+${f.vHead})`} w={210} tone="tint" />

      <Arrow x1={60} y1={208} x2={60} y2={230} />
      <Box x={0} y={230} label="K_nope · V" sub={`${f.heads} heads × ${f.qkNope} / ${f.vHead}`} w={210} />
      <Arrow x1={265} y1={166} x2={265} y2={230} />
      <text x={272} y={200} className="blocksub" fill="var(--ember)">RoPE on q_pe, k_pe</text>

      <Arrow x1={105} y1={264} x2={105} y2={286} />
      <Arrow x1={372} y1={166} x2={372} y2={286} />
      <Box x={40} y={286} label="softmax(QKᵀ / √d + causal)" sub={`${f.heads} × T × T`} w={355} tone="ember" />
      <Arrow x1={217} y1={320} x2={217} y2={344} />
      <Box x={80} y={344} label="attn ⊗ V" sub={`T × ${f.heads * f.vHead}`} w={275} />
      <Arrow x1={217} y1={378} x2={217} y2={400} />
      <Box x={120} y={400} label="W_o" sub={`${f.heads * f.vHead} → ${f.hidden}`} w={200} />
      <Arrow x1={220} y1={434} x2={220} y2={458} />
      <Box x={130} y={458} label="+ residual" sub={`T × ${f.hidden}`} w={180} tone="ember" />
      <text x={0} y={510} className="blocksub">
        {f.qLora ? "Q is compressed too (q_lora_rank set). " : "Q is projected directly; V2-Lite does not compress queries. "}
        KV is stored as {kvOut + f.qkRope} channels per token.
      </text>
    </svg>
  );
}

function MoeDiagram({ f }: { f: ArchFocus }) {
  const W = 520, H = 520;
  return (
    <svg className="arch-svg" viewBox={`0 0 ${W} ${H}`} role="img">
      <text x={0} y={10} className="mini-label" style={{ fontSize: 8, letterSpacing: "0.1em" }}>SUBBLOCK 2 · DEEPSEEKMOE</text>
      <Box x={0} y={22} label="residual x" sub={`T × ${f.hidden}`} w={135} />
      <Arrow x1={135} y1={39} x2={175} y2={39} />
      <Box x={175} y={22} label="RMSNorm" sub={`${f.hidden} gain`} w={100} />
      <Arrow x1={275} y1={39} x2={310} y2={39} />
      <Box x={310} y={22} label="gate W_g" sub={`${f.hidden} × ${f.experts}`} w={115} tone="tint" />
      <Arrow x1={367} y1={56} x2={367} y2={78} />
      <Box x={280} y={78} label={`top-${f.topk} of ${f.experts}`} sub={`scores → weights`} w={145} tone="ember" />

      <Arrow x1={225} y1={56} x2={225} y2={78} />
      <Box x={0} y={78} label={`${f.experts} routed experts`} sub={`each 3 × ${f.hidden} × ${f.moeInter}`} w={225} />
      <Box x={0} y={122} label="SwiGLU expert" sub={`${f.hidden} → ${f.moeInter} → ${f.hidden}`} w={225} tone="tint" />
      <Arrow x1={112} y1={156} x2={112} y2={178} />
      <Box x={0} y={178} label={`Σ over selected experts`} sub={`T × ${f.hidden} · scaling ${f.routedScale}`} w={225} />
      <Arrow x1={112} y1={212} x2={112} y2={240} />
      <Box x={0} y={240} label="+ shared expert" sub={`${f.shared} × ${f.moeInter} wide, always on`} w={225} tone="ember" />
      <Arrow x1={112} y1={274} x2={112} y2={298} />
      <Box x={0} y={298} label="+ residual" sub={`T × ${f.hidden}`} w={225} tone="ember" />
      <Arrow x1={352} y1={112} x2={352} y2={298} />
      <Box x={280} y={298} label="gating weights" sub={`T × ${f.topk}`} w={145} />

      <text x={0} y={344} className="mini-label" style={{ fontSize: 8 }}>ROUTING</text>
      <Box x={0} y={354} label="scoring" sub={f.scoring} w={135} />
      <Box x={145} y={354} label="selection" sub={f.topkMethod} w={135} />
      <Box x={290} y={354} label="normalise" sub={f.normTopk ? "top-k sums to 1" : "× scaling factor"} w={135} />
      <text x={0} y={414} className="blocksub">
        Dense prefix: {f.firstDense} layer{f.firstDense === 1 ? "" : "s"}. Hash-routed: {f.hashLayers} layer{f.hashLayers === 1 ? "" : "s"}.
      </text>
      <text x={0} y={430} className="blocksub">
        Group-limited routing: {f.nGroups ?? 1} group{f.nGroups === 1 ? "" : "s"}, top-{f.topkGroup ?? 1}.
      </text>
      <text x={0} y={452} className="blocksub" fill="var(--ember)">
        Every routed expert stores 3 × {f.hidden} × {f.moeInter} = {compact(3 * f.hidden * f.moeInter, 2)} parameters.
      </text>
      <text x={0} y={474} className="blocksub">
        Active FFN per token: {f.topk} experts + {f.shared} shared = {compact((f.topk + f.shared) * 3 * f.hidden * f.moeInter, 2)} parameters.
      </text>
      <text x={0} y={496} className="blocksub">
        Total FFN capacity per layer: {compact((f.experts + f.shared) * 3 * f.hidden * f.moeInter, 1)}.
      </text>
    </svg>
  );
}

function LayerStrip({ f, onHover }: { f: ArchFocus; onHover: (p: LayerPlan | null) => void }) {
  const { ref, width } = useMeasure<HTMLDivElement>();
  const L = f.plan.length;
  const cw = Math.max(7, Math.min(22, (width - 20) / L));
  const h = 122;
  const x = (l: number) => 12 + l * cw;
  const color = (p: LayerPlan) =>
    p.attn === "hca" ? "#15246e" : p.attn === "csa" ? "#2b46f0" : p.attn === "dsa" ? "#6a3f73" : p.attn === "sliding" ? "#8491bb" : "#b58a4a";
  return (
    <div ref={ref} style={{ overflowX: "auto" }}>
      <svg width={Math.max(width, 12 + L * cw + 12)} height={h} className="chart">
        {Array.from({ length: L }, (_, l) => {
          const p = f.plan[l];
          return (
            <g key={l} onMouseEnter={() => onHover(p)} onMouseLeave={() => onHover(null)}>
              <rect x={x(l)} y={18} width={cw - 2} height={34} fill={color(p)} opacity={0.9} />
              <rect x={x(l)} y={58} width={cw - 2} height={20} fill={p.ffn === "moe" ? "#101217" : "#bcb7a9"} />
              {p.hash && <circle cx={x(l) + (cw - 2) / 2} cy={68} r={3} fill="var(--ember)" />}
              {p.kv_source && <circle cx={x(l) + (cw - 2) / 2} cy={92} r={4} fill="none" stroke="var(--teal)" strokeWidth={1.5} />}
              {p.kv_shared && <line x1={x(l) + 2} y1={92} x2={x(l) + cw - 4} y2={92} stroke="var(--rule-2)" />}
              <line x1={x(l) + cw / 2 - 1} y1={100} x2={x(l) + cw / 2 - 1} y2={104} stroke="var(--rule-2)" />
              {l % 5 === 0 && <text x={x(l) + cw / 2} y={116} textAnchor="middle" className="ticklabel">{l}</text>}
            </g>
          );
        })}
        <text x={0} y={12} className="ticklabel">attn</text>
        <text x={0} y={70} className="ticklabel">FFN</text>
        <text x={0} y={95} className="ticklabel">KV</text>
      </svg>
    </div>
  );
}

export default function Architecture() {
  const archQ = useOne<ArchData>("architecture");
  const bQ = useJson<Budget>("budget");
  const [sel, setSel] = useState<ModelKey>("v4flash");
  const [hover, setHover] = useState<LayerPlan | null>(null);
  const arch = archQ.data;
  const f = arch?.focus?.[sel];
  const v3 = arch?.focus.v3;
  const v4flash = arch?.focus.v4flash;
  const v41 = arch?.focus.v41flash;
  const v41Sources = v41?.plan.filter((p) => p.kv_source).length ?? 0;
  const v41Readers = (v41?.layers ?? 0) - v41Sources;
  const budget = bQ.data?.budget?.members.find((m) => m.id === sel);

  const total = budget?.total ?? 0;
  const shared = budget ? budget.sharedPerLayer * Math.max(0, budget.layers - (f?.firstDense ?? 0)) : 0;
  const routed = Math.max(0, (budget?.moe ?? 0) - shared);
  const parts = budget
    ? [
        { k: "routed experts", v: routed, c: "var(--signal)" },
        { k: "shared expert", v: shared, c: "var(--ember)" },
        { k: "attention", v: budget.attn, c: "var(--teal)" },
        { k: "dense FFN", v: budget.dense, c: "var(--sand)" },
        { k: "embeddings", v: budget.embed + budget.lmHead, c: "var(--plum)" },
        { k: "norms", v: budget.norms, c: "var(--ink-3)" },
      ]
    : [];

  return (
    <>
      <ChapterHead
        n="02" part="I" kicker="Architecture"
        title={<>The block, in <em>three</em> generations</>}
        standfirst={
          <>
            The family's block is a residual stream with two unusual sublayers: an
            attention that reads and writes through a compressed latent, and an FFN
            that is mostly dormant. Switch checkpoint to watch the same diagram
            acquire query compression, hash routing, compressed attention and MTP.
          </>
        }
        chips={<><Chip tone="blue">config-derived</Chip><Chip>{f?.layers ?? "—"} layers</Chip><Chip tone="teal">{f?.experts || "dense"} experts</Chip></>}
      />

      <div className="row" style={{ marginBottom: 4 }}>
        <Seg value={sel} options={MODELS.map((m) => ({ v: m.v, label: m.label }))} onChange={setSel} />
        <div style={{ flex: 1 }} />
        <span className="mono" style={{ fontSize: 9.5, color: "var(--ink-3)" }}>
          d={f?.hidden} · heads={f?.heads} · qk={f ? f.qkNope + f.qkRope : "—"} · v={f?.vHead}
        </span>
      </div>

      <DataState loading={archQ.loading} error={archQ.error} what="the architecture configs">
        {f && (
          <>
            <div className="split" style={{ gap: 18 }}>
              <Panel
                title={<><b>Figure 2.1</b> · the two sublayers, with named shapes</>}
                tight
                note="Every box is a named matrix or a measured tensor shape from the checkpoint's config. The residual is carried around both sublayers; the diagram omits dropout, which is zero at inference."
              >
                <div className="arch-pair">
                  <MlaDiagram f={f} />
                  <MoeDiagram f={f} />
                </div>
              </Panel>
            </div>

            <Panel
              title={<><b>Figure 2.2</b> · the depth plan</>}
              tools={<Legend items={[
                { c: "#2b46f0", label: "CSA / DSA" },
                { c: "#15246e", label: "HCA 128:1" },
                { c: "#b58a4a", label: "MLA" },
                { c: "#101217", label: "MoE" },
              ]} />}
              note="Each column is one block. Top band: attention type. Middle band: dense (pale) or MoE (ink); an ember dot marks a hash-routed layer. Bottom: a teal ring marks a layer that stores KV for others; a line means it reads a shared cache."
            >
              <LayerStrip f={f} onHover={setHover} />
              <div className="hoverbox" style={{ marginTop: 8 }}>
                {hover
                  ? <>layer <b>{hover.layer}</b> · {hover.attn.toUpperCase()} attention · {hover.ffn} FFN · compress {hover.compress || 1}× {hover.hash ? "· hash-routed" : ""} {hover.kv_source ? "· KV source" : hover.kv_shared ? "· shared KV" : ""}</>
                  : "Hover a block to read its role. The plan is read directly from config fields (compress_ratios, num_hash_layers, kv_source_layer_ids, first_k_dense_replace)."}
              </div>
            </Panel>

            <h2 className="sec">Where the parameters are</h2>
            <Panel
              title={<><b>Figure 2.3</b> · parameter budget of {f.name}</>}
              tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>{compact(total, 1)} total</span>}
              note="Computed from tensor shapes, not copied from a table. The routed experts are the vast majority of FFN capacity but only a small fraction of what a token pays for.">
              <div className="bar-stack">
                {parts.map((p) => (
                  <span key={p.k} style={{ width: `${(p.v / total) * 100}%`, background: p.c }}
                    title={`${p.k}: ${compact(p.v)} (${fmt((p.v / total) * 100, 2)}%)`} />
                ))}
              </div>
              <div className="grid3" style={{ marginTop: 14 }}>
                <Stat k="Routed experts" v={`${fmt((routed / total) * 100, 1)}%`} s={`${f.experts} × ${compact(3 * f.hidden * f.moeInter, 2)} each`} />
                <Stat k="Active experts" v={`${fmt(((f.topk + f.shared) / (f.experts + f.shared)) * 100, 1)}%`} s={`${f.topk} of ${f.experts} routed + ${f.shared} shared`} />
                <Stat k="Attention" v={`${fmt(((budget?.attn ?? 0) / total) * 100, 1)}%`} s={f.qLora ? "Q and KV both compressed" : "KV compressed, Q direct"} />
              </div>
            </Panel>

            <Prose>
              <p>
                Two details in this diagram carry most of the family's behaviour. First,
                the <strong>KV write is a bottleneck</strong>: a token is represented in
                the cache by {f.kvLora ?? f.headDim} latent channels plus {f.qkRope}{" "}
                positional channels, no matter how many query heads read it. Attention
                can be wide without a wide cache. Second, the <strong>FFN is a routing
                problem</strong>: {compact((f.experts + f.shared) * 3 * f.hidden * f.moeInter, 1)}{" "}
                parameters exist in this block, but only{" "}
                {compact((f.topk + f.shared) * 3 * f.hidden * f.moeInter, 2)} are applied
                to a token. The gate is {compact(f.hidden * f.experts, 0)} parameters
                deciding the fate of {compact(f.experts * 3 * f.hidden * f.moeInter, 1)}.
              </p>
              <p>
                The generation differences are visible as accretion. V2-Lite projects Q
                directly; V3 compresses Q through a {v3?.qLora}-rank latent. V4 replaces
                the all-MLA stack with alternating compressed-sparse and heavily
                compressed layers; V4-Flash hashes its first {v4flash?.hashLayers} MoE
                layers instead of learning a gate, while V4.1 hashes none. V4.1 removes
                most KV stores entirely: {v41Sources} of its {v41?.layers} layers hold
                the cache that the other {v41Readers} read.
              </p>
            </Prose>

            <div className="grid4">
              <Stat k="Residual width" v={f.hidden} s={`${f.heads} query heads`} />
              <Stat k="Q latent" v={f.qLora ?? "direct"} s={f.qLora ? "low-rank query path" : "no query compression"} />
              <Stat k="KV latent" v={f.kvLora ?? f.headDim} s={`+ ${f.qkRope} RoPE channels`} />
              <Stat k="MTP heads" v={f.mtp || "—"} s={f.mtp ? "draft blocks, excluded from totals" : "none"} />
            </div>

            <Method label="How the block diagram is generated">
              <p>
                Shapes are products of config fields: <code>hidden_size</code>,{" "}
                <code>num_attention_heads</code>, <code>qk_nope_head_dim</code>,{" "}
                <code>qk_rope_head_dim</code>, <code>v_head_dim</code>,{" "}
                <code>kv_lora_rank</code>, <code>moe_intermediate_size</code>,{" "}
                <code>n_routed_experts</code>, <code>n_shared_experts</code>,{" "}
                <code>num_experts_per_tok</code>. No shape is illustrative.
              </p>
              <p>
                V4.1 nests these fields under <code>text_config</code>; the diagram
                reads that block. Its vision encoder and projector are outside the
                language block and are not drawn.
              </p>
            </Method>

            <Footnotes
              items={[
                "The block is pre-norm: RMSNorm is applied inside each sublayer to its input, and the sublayer output is added to the unmodified residual. The diagram follows the checkpoint code, not a generic transformer.",
                "MLA's latent is not a low-rank approximation of the attention matrix; it is a low-rank parameterisation of the K/V projections. The model still computes full-rank per-head keys and values after the up-projection.",
                "Hash routing in V4 assigns an expert by a fixed hash of the token id in the first layers. It removes the learned gate, not the expert computation; the layer remains an MoE.",
                "MTP blocks are separate transformer blocks with their own norms and projections. Published total-parameter counts exclude them; the architecture chapter does too.",
              ]}
            />
          </>
        )}
      </DataState>
      <PageFoot index={2} />
    </>
  );
}
