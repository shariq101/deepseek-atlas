import React, { useMemo, useState } from "react";
import {
  Architecture as ArchData, ArchFocus, Family as FamilyData, FamilyMember, LayerPlan,
  compact, fmt, useOne,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";

const ATTN_COLORS: Record<string, string> = {
  mla: "#b58a4a",
  dsa: "#6a3f73",
  csa: "#2b46f0",
  hca: "#15246e",
  sliding: "#8491bb",
};

function MemberBlock({ m }: { m: FamilyMember }) {
  const W = 760;
  const L = m.plan.length || m.layers;
  const left = 78, right = 20;
  const cw = Math.max(4, Math.min(16, (W - left - right) / Math.max(L, 1)));
  const x = (l: number) => left + l * cw;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={W} height={220} viewBox={`0 0 ${W} 220`} style={{ width: W, minWidth: W, height: 220 }} className="chart" role="img" aria-label={`Block plan for ${m.name}`}>
        <text x={22} y={16} className="mini-label">ATTENTION TYPE · FFN TYPE · KV SOURCE · HASH LAYER</text>
        <line x1={left} y1={146} x2={W - right} y2={146} stroke="var(--signal)" strokeWidth={1.4} />
        <text x={22} y={164} className="blocksub">residual stream</text>
        {Array.from({ length: L }, (_, l) => {
          const p: LayerPlan | undefined = m.plan[l];
          if (!p) return null;
          const on = p.kv_source;
          return (
            <g key={l}>
              <rect x={x(l)} y={42} width={Math.max(3, cw - 1.2)} height={30} fill={ATTN_COLORS[p.attn] ?? "var(--ink-3)"} opacity={0.9} />
              <rect x={x(l)} y={78} width={Math.max(3, cw - 1.2)} height={24} fill={p.ffn === "moe" ? "var(--ink)" : "var(--rule-2)"} opacity={p.ffn === "moe" ? 0.9 : 0.6} />
              {p.hash && <circle cx={x(l) + (cw - 1.2) / 2} cy={90} r={2.2} fill="var(--ember)" />}
              {on && <circle cx={x(l) + (cw - 1.2) / 2} cy={118} r={3.2} fill="none" stroke="var(--teal)" strokeWidth={1.4} />}
              {p.kv_shared && <line x1={x(l)} y1={118} x2={x(l) + cw - 1.2} y2={118} stroke="var(--rule-2)" />}
              {(l % 5 === 0 || l === L - 1) && <text x={x(l) + cw / 2} y={192} textAnchor="middle" className="ticklabel">{l}</text>}
            </g>
          );
        })}
        <text x={22} y={36} className="blocksub">attention</text>
        <text x={22} y={84} className="blocksub">FFN</text>
        <text x={22} y={122} className="blocksub">KV source</text>
        <text x={W - 22} y={210} textAnchor="end" className="blocksub">{m.attention} · {m.experts ? `${m.experts} routed · top-${m.topk}` : "dense FFN"}</text>
      </svg>
    </div>
  );
}

function ArchBox({ x, y, w, h, title, lines, tone = "" }: {
  x: number; y: number; w: number; h: number; title: string; lines: string[]; tone?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} className={"blockbox " + tone} rx={1} />
      <text x={x + 8} y={y + 14} className="blocklabel">{title}</text>
      {lines.map((line, i) => (
        <text key={line} x={x + 8} y={y + 26 + i * 10} className="blocksub">{line}</text>
      ))}
    </g>
  );
}

function MemberArchitecture({ m, f }: { m: FamilyMember; f?: ArchFocus }) {
  const W = 760, H = 334;
  const densePrefix = f?.firstDense ?? m.plan.filter((p) => p.ffn === "dense").length;
  const q = f ? (f.qLora ?? "direct") : "direct";
  const kv = f ? (f.kvLora ?? f.headDim) : m.headDim;
  const ffn = m.experts ? `${m.experts} routed · top-${m.topk} · ${m.shared} shared` : "dense FFN";
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} className="chart" role="img" aria-label={`Architecture of ${m.name}`}>
        <text x={22} y={16} className="mini-label">ARCHITECTURE OF {m.name.toUpperCase()} · CONFIG-DERIVED</text>

        <line x1={180} y1={117} x2={180} y2={282} stroke="var(--signal)" strokeWidth={2} />
        <line x1={180} y1={142} x2={200} y2={142} stroke="var(--signal)" strokeWidth={1.2} />
        <line x1={180} y1={230} x2={200} y2={230} stroke="var(--signal)" strokeWidth={1.2} />
        <line x1={180} y1={279} x2={200} y2={279} stroke="var(--signal)" strokeWidth={1.2} />
        <text x={22} y={300} className="blocksub" fill="var(--signal)">residual stream</text>

        <ArchBox x={20} y={42} w={150} h={38} title="token + position" lines={[`T × ${m.hidden}`, "input sequence"]} tone="tint" />
        <line x1={95} y1={80} x2={95} y2={100} stroke="var(--ink-3)" strokeWidth={1.1} />
        <ArchBox x={20} y={100} w={150} h={34} title="RMSNorm" lines={["pre-norm"]} />
        <line x1={170} y1={117} x2={200} y2={117} stroke="var(--ink-3)" strokeWidth={1.1} />

        <ArchBox x={200} y={92} w={210} h={50} title={`attention · ${m.attention}`} lines={[`${m.heads} heads · q ${q}`, `kv ${kv} channels · ${m.kv.scheme}`]} tone="tint" />
        <line x1={410} y1={117} x2={440} y2={117} stroke="var(--ink-3)" strokeWidth={1.1} />
        <ArchBox x={440} y={92} w={150} h={50} title="KV cache" lines={[`${fmt(m.kv.per_token, 1)} B / token`, `${m.kv.layers_stored} stored layers`]} tone="ember" />

        <line x1={305} y1={142} x2={305} y2={174} stroke="var(--ink-3)" strokeWidth={1.1} />
        <ArchBox x={200} y={174} w={210} h={56} title={`FFN · ${m.experts ? "MoE" : "dense"}`} lines={[ffn, `dense prefix ${densePrefix} · width ${m.moeInter || "—"}`]} tone="ember" />
        <line x1={410} y1={202} x2={440} y2={202} stroke="var(--ink-3)" strokeWidth={1.1} />
        <ArchBox x={440} y={176} w={150} h={52} title="shared + routed" lines={[m.experts ? `${m.shared} shared · ${m.topk} routed` : "one dense FFN", m.experts ? "weighted residual write" : "residual write"]} />

        <line x1={305} y1={230} x2={305} y2={258} stroke="var(--ink-3)" strokeWidth={1.1} />
        <ArchBox x={200} y={258} w={210} h={42} title="final RMSNorm + LM head" lines={[`vocab ${compact(m.vocab, 0)} · MTP ${m.mtp || "none"}`, "next-token logits"]} />

        <ArchBox x={440} y={242} w={150} h={58} title="context + precision" lines={[m.context ? `${compact(m.context, 0)} tokens` : "context —", m.precision]} />
        <ArchBox x={610} y={92} w={128} h={50} title="block facts" lines={[`${m.layers} layers`, `${m.hash_layers || 0} hash`]} tone="tint" />

        <text x={22} y={322} className="blocksub">Q and KV are the attention paths; FFN is the routed/shared feed-forward path.</text>
      </svg>
    </div>
  );
}

export default function FamilyBlocksPage() {
  const q = useOne<FamilyData>("family");
  const archQ = useOne<ArchData>("architecture");
  const members = q.data?.members ?? [];
  const [id, setId] = useState("v4flash");
  const m = members.find((x) => x.id === id) ?? members[0];
  const focus = m ? archQ.data?.focus?.[m.id] : undefined;
  const totals = useMemo(() => ({
    layers: m?.layers ?? 0,
    experts: m?.experts ?? 0,
    active: m?.published_active ?? 0,
    total: m?.published_total ?? 0,
  }), [m]);

  return (
    <>
      <ChapterHead n="35" part="V" kicker="Theory · member blocks"
        title={<>Every member, <em>as a block</em></>}
        standfirst={<>The family pages compare numbers. This page keeps the block fixed and swaps the checkpoint underneath it: attention scheme, expert count, top-k, dense prefix, hash layers and cache sharing. Pick a member to see where the same skeleton differs.</>}
        chips={<><Chip tone="blue">config-derived</Chip><Chip>{members.length || 9} members</Chip><Chip tone="teal">same block, different constants</Chip></>}
      />

      <DataState loading={q.loading} error={q.error} what="the family configs">
        {m && (
          <>
            <div className="row" style={{ gap: 10, marginBottom: 10 }}>
              <span className="mini-label">checkpoint</span>
              <select className="ctl" value={m.id} onChange={(e) => setId(e.target.value)} style={{ maxWidth: 260 }}>
                {members.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
              </select>
              <span className="caption">{m.attention} · {m.layers} layers · {m.context ? `${compact(m.context, 0)} context` : "context —"}</span>
            </div>

            <div className="grid4" style={{ marginBottom: 16 }}>
              <Stat k="Stored" v={compact(totals.total)} s={m.name} />
              <Stat k="Active" v={compact(totals.active)} s={`${fmt(m.sparsity, 1)}× sparsity`} />
              <Stat k="Routed experts" v={m.experts ? String(m.experts) : "dense"} s={m.experts ? `top-${m.topk} · ${m.shared} shared` : "one dense FFN"} />
              <Stat k="Depth" v={String(m.layers)} s={`${m.plan.filter((p) => p.ffn === "dense").length} dense block${m.plan.filter((p) => p.ffn === "dense").length === 1 ? "" : "s"} in plan`} />
            </div>

            <Panel title={<><b>Figure 35.1</b> · the block plan of {m.name}</>}
              note="Attention colour: MLA sand, DSA plum, CSA blue, HCA deep blue, sliding blue-grey. FFN: pale is dense, ink is MoE. Ember dot: hash-routed layer. Teal ring: layer that stores a shared KV cache. The plan is read from config fields; it is not a measured routing trace.">
              <MemberBlock m={m} />
            </Panel>

            <Panel title={<><b>Figure 35.2</b> · architecture of {m.name}</>}
              note="A config-derived block map: pre-norm attention, the KV path, the feed-forward path, and the final projection. The shape labels come from family.json and architecture.json; no forward pass is run on this page.">
              <MemberArchitecture m={m} f={focus} />
            </Panel>

            <div className="grid2" style={{ alignItems: "start" }}>
              <Panel title={<><b>Figure 35.3</b> · what is specific to this member</>} tight>
                <table className="tbl">
                  <tbody>
                    <tr><td className="k">attention</td><td>{m.attention}</td></tr>
                    <tr><td className="k">Q / KV latent</td><td>{focus?.qLora ?? "direct"} / {focus?.kvLora ?? focus?.headDim ?? m.headDim}</td></tr>
                    <tr><td className="k">layers</td><td>{m.layers}</td></tr>
                    <tr><td className="k">hidden / heads</td><td>{m.hidden} / {m.heads}</td></tr>
                    <tr><td className="k">experts</td><td>{m.experts || "dense"} {m.experts ? `· top-${m.topk} · ${m.shared} shared` : ""}</td></tr>
                    <tr><td className="k">expert width</td><td>{m.moeInter || "—"}</td></tr>
                    <tr><td className="k">dense prefix</td><td>{focus?.firstDense ?? m.plan.filter((p) => p.ffn === "dense").length}</td></tr>
                    <tr><td className="k">context</td><td>{m.context ? compact(m.context, 0) : "—"}</td></tr>
                    <tr><td className="k">precision</td><td>{m.precision}</td></tr>
                    <tr><td className="k">MTP heads</td><td>{m.mtp || "—"}</td></tr>
                    <tr><td className="k">hash layers</td><td>{m.hash_layers || "—"}</td></tr>
                    <tr><td className="k">KV / token</td><td>{m.kv?.per_token ? `${fmt(m.kv.per_token, 1)} B` : "—"}</td></tr>
                  </tbody>
                </table>
              </Panel>
              <Panel title={<><b>Figure 35.4</b> · the same block across the family</>} tight>
                <div className="scroll" style={{ maxHeight: 320 }}>
                  <table className="tbl">
                    <thead><tr><th>member</th><th>layers</th><th>experts</th><th>top-k</th><th>attention</th></tr></thead>
                    <tbody>
                      {members.map((x) => (
                        <tr key={x.id} className={x.id === m.id ? "hi" : ""} onClick={() => setId(x.id)} style={{ cursor: "pointer" }}>
                          <td className="k">{x.short}</td><td>{x.layers}</td><td>{x.experts || "—"}</td>
                          <td>{x.topk || "—"}</td><td>{x.attention}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </div>

            <Prose>
              <p className="aside">
                <strong>Read this page as a shape, not a trace.</strong> The plan shows what
                the config says each layer is. It does not say which experts fired for a
                token, whether a hash layer was active, or how a cache source was used in a
                particular forward pass. Those are measured elsewhere; this page is the
                architectural key that makes those measurements comparable.
              </p>
            </Prose>
          </>
        )}
      </DataState>

      <Method label="How the member blocks are drawn">
        <p>
          The block plan comes from the same family.json used by chapters 1 and 2. Each
          layer's attention and FFN type comes from the config-derived plan; hash and
          KV-source markers come from config fields. The architecture panel also reads
          architecture.json for Q/KV ranks, scoring and dense-prefix fields. No weights are
          run on this page.
        </p>
      </Method>
      <Footnotes items={[
        "The block plan and architecture map are config-derived, not a complete implementation description. Tensor shapes and cache arithmetic are in chapters 2–4.",
        "R1 shares V3's architecture. The plan is shown as a control branch, not a separate block design.",
        "V4.1's 40 layers are represented by its published encoder/decoder plan; no V4.1 forward pass exists in this atlas.",
      ]} />
      <PageFoot index={35} />
    </>
  );
}
