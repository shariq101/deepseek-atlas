import React, { useMemo, useState } from "react";
import {
  Cache as CacheData, CacheMember, bytes, compact, fmt, pct, useMeasure, useOne,
} from "../lib/atlas";
import {
  Caption, ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { LineChart, Series } from "../components/charts";

const SCHEMES = [
  {
    name: "MHA", tag: "the reference",
    formula: "2 · L · H · d_h · bytes",
    note: "Every head stores its own K and V. The cache grows with heads.",
    bars: 18, width: 4, color: "var(--ink-3)",
  },
  {
    name: "GQA", tag: "shared KV heads",
    formula: "2 · L · H_kv · d_h · bytes",
    note: "Query heads share a smaller set of KV heads. Cheaper, but still per-head width.",
    bars: 6, width: 7, color: "var(--teal)",
  },
  {
    name: "MLA", tag: "latent KV",
    formula: "L · (kv_lora + rope) · bytes",
    note: "One latent per token per layer, plus a decoupled RoPE key. Heads read it through an up-projection.",
    bars: 2, width: 11, color: "var(--signal)",
  },
  {
    name: "CSA / HCA", tag: "compressed + shared",
    formula: "Σ_l (d_head + rope) / r_l · bytes",
    note: "Compress along the sequence, store only source layers, let the rest read a shared cache.",
    bars: 3, width: 5, color: "var(--ember)",
  },
];

function SchemeCard({ s }: { s: (typeof SCHEMES)[number] }) {
  return (
    <div style={{ borderTop: "1px solid var(--ink)", paddingTop: 8 }}>
      <div className="spread">
        <span style={{ fontFamily: "var(--serif)", fontSize: 21 }}>{s.name}</span>
        <span className="mini-label">{s.tag}</span>
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 44, margin: "8px 0 7px" }}>
        {Array.from({ length: s.bars }, (_, i) => (
          <span key={i} style={{ width: s.width, height: 18 + (i % 4) * 6, background: s.color, opacity: 0.85 }} />
        ))}
      </div>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--ember)", marginBottom: 5 }}>{s.formula}</div>
      <div style={{ fontSize: 11, color: "var(--ink-3)", lineHeight: 1.55 }}>{s.note}</div>
    </div>
  );
}

function BytesDotPlot({ members }: { members: CacheMember[] }) {
  const { ref, width } = useMeasure<HTMLDivElement>();
  const h = 290;
  const m = { t: 20, r: 108, b: 32, l: 60 };
  const lo = Math.log10(2 ** 10), hi = Math.log10(2 ** 17);
  const sx = (v: number) => m.l + ((Math.log10(v) - lo) / (hi - lo)) * (width - m.l - m.r);
  const rowH = (h - m.t - m.b) / members.length;
  const ticks = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16];
  const color = (x: CacheMember) => x.scheme === "MLA" ? "var(--signal)" : x.scheme === "CSA/HCA" ? "var(--ember)" : "var(--teal)";
  return (
    <div ref={ref} style={{ overflowX: "auto" }}>
      <svg width={width} height={h} className="chart">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.t - 6} y2={h - m.b} className="gridline" />
            <text x={sx(t)} y={h - m.b + 14} textAnchor="middle" className="ticklabel">{bytes(t)}</text>
          </g>
        ))}
        {members.map((x, i) => {
          const y = m.t + rowH * (i + 0.5);
          return (
            <g key={x.id}>
              <line x1={m.l} x2={sx(x.perToken)} y1={y} y2={y} stroke="var(--rule)" strokeWidth={0.8} />
              <circle cx={sx(x.perToken)} cy={y} r={4.5} fill={color(x)} />
              <text x={m.l - 8} y={y + 3} textAnchor="end" className="ticklabel" style={{ fill: "var(--ink-2)" }}>{x.short}</text>
              <text x={sx(x.perToken) + 9} y={y + 3} className="ticklabel" style={{ fill: "var(--ink-2)" }}>
                {bytes(x.perToken)}
              </text>
            </g>
          );
        })}
        <text x={m.l} y={12} className="axislabel">bytes per token, fp16 · log →</text>
      </svg>
      <div className="hoverbox">Cache per token, one sequence. V4's figure is a model: only layers that store KV are counted, and each is divided by its compression ratio.</div>
    </div>
  );
}

function CompressStrip({ m }: { m: CacheMember }) {
  const parts = m.compress ?? [];
  const L = parts.length || m.layers;
  const cw = Math.max(6, Math.min(22, 900 / L));
  const x = (l: number) => 20 + l * cw;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={Math.max(920, 20 + L * cw)} height={108} className="chart">
        {Array.from({ length: L }, (_, l) => {
          const r = parts[l] ?? 1;
          const col = r >= 64 ? "#15246e" : r > 0 ? "#2b46f0" : "#d9d5c8";
          const src = (m.kvSource ?? []).includes(l);
          return (
            <g key={l}>
              <rect x={x(l)} y={20} width={cw - 1.5} height={34} fill={col} opacity={r ? 0.9 : 0.5} />
              <text x={x(l) + (cw - 1.5) / 2} y={42} textAnchor="middle"
                style={{ fontFamily: "var(--mono)", fontSize: 7, fill: r >= 64 ? "#fff" : r > 0 ? "#fff" : "var(--ink-3)" }}>
                {r ? (r === 4 ? "4" : r === 128 ? "128" : String(r)) : "—"}
              </text>
              {src && <circle cx={x(l) + (cw - 1.5) / 2} cy={67} r={4} fill="none" stroke="var(--teal)" strokeWidth={1.5} />}
              {l % 5 === 0 && <text x={x(l) + cw / 2} y={92} textAnchor="middle" className="ticklabel">{l}</text>}
            </g>
          );
        })}
        <text x={0} y={12} className="axislabel">per-layer compression ratio · ring = stores KV</text>
      </svg>
    </div>
  );
}

export default function Cache() {
  const { data, loading, error } = useOne<CacheData>("cache");
  const [sel, setSel] = useState("v4flash");
  const members = data?.members ?? [];
  const m = members.find((x) => x.id === sel);
  const mla = members.find((x) => x.id === "v3") ?? members.find((x) => x.kvLora);
  const { ref, width } = useMeasure<HTMLDivElement>();

  const series = useMemo(() => {
    const ctxs = data?.contexts ?? [];
    const pick = ["v3", "v4flash", "v41flash"];
    const lines: Series[] = pick.flatMap((id) => {
      const x = members.find((y) => y.id === id);
      if (!x) return [];
      return [{
        color: id === "v3" ? "var(--signal)" : id === "v4flash" ? "var(--ember)" : "var(--teal)",
        label: x.short, width: 2,
        values: ctxs.map((c) => Math.log10(x.at[String(c)] ?? 1)),
      }];
    });
    const v3 = members.find((x) => x.id === "v3");
    if (v3) {
      lines.push({
        color: "var(--rule-2)", label: "MHA at V3 width", dash: "3 4", width: 1.4,
        values: (data?.contexts ?? []).map((c) => Math.log10(v3.mha * c)),
      });
    }
    return lines;
  }, [members, data]);

  const ctxs = data?.contexts ?? [];

  return (
    <>
      <ChapterHead
        n="04" part="I" kicker="KV cache"
        title={<>The compression of <em>memory</em></>}
        standfirst={
          <>
            A long-context model is usually killed by its cache, not by its weights.
            DeepSeek's answer has evolved from a low-rank latent to sequence compression
            and cross-layer sharing. This chapter makes the bytes visible, because the
            bytes are the architecture.
          </>
        }
        chips={<><Chip tone="blue">config arithmetic</Chip><Chip>fp16, one sequence</Chip><Chip tone="teal">437× claimed at V4.1</Chip></>}
      />

      <DataState loading={loading} error={error} what="the cache arithmetic">
        <div className="grid4">
          {SCHEMES.map((s) => <SchemeCard key={s.name} s={s} />)}
        </div>

        <Panel
          title={<><b>Figure 4.1</b> · bytes per token, every member</>}
          note="Logged axis, FP16. A 1,024-token cache is about the size of a small image; a 1M-token cache is the difference between a workstation and a datacenter. MLA buys 7–57× against MHA at the same width before any sequence compression. V4.1's published 890 B/token is FP4 and sits below this axis; it is reported separately in Figure 4.4."
        >
          <BytesDotPlot members={members} />
        </Panel>

        <div className="split">
          <Panel
            title={<><b>Figure 4.2</b> · cache growth with context</>}
            tools={
              <select className="ctl" value={sel} onChange={(e) => setSel(e.target.value)} style={{ maxWidth: 160 }}>
                {members.filter((x) => x.compress).map((x) => <option key={x.id} value={x.id}>{x.short}</option>)}
              </select>
            }
            note="Both axes are logarithmic. The dashed line is multi-head attention at V3's width: it is not a deployed option, it is the cost the family is designed to avoid. V4.1's line is the flattest because it both compresses and shares."
          >
            <div ref={ref} style={{ overflowX: "auto" }}>
              <LineChart
                width={Math.min(Math.max(width, 360), 760)} height={300}
                series={series}
                yDomain={[3, 13.2]}
                xTicks={[0, 1, 2, 3, 4, 5]}
                xFormat={(i) => compact(ctxs[Math.round(i)] ?? 0, 0)}
                yTicks={[4, 6, 8, 10, 12]}
                yFormat={(v) => bytes(10 ** v)}
                xLabel="context length" yLabel="cache for one sequence"
              />
            </div>
            <div className="legend" style={{ marginTop: 8 }}>
              <Legend items={series.map((s) => ({ c: s.color, label: s.label ?? "", line: true }))} />
            </div>
          </Panel>

          <div className="stack">
            <Panel
              title={<><b>Figure 4.3</b> · what one million tokens costs</>}
              note="Total cache for a one-million-token sequence, computed per member (not necessarily its native maximum). V3.1 and V3.2 share V3's MLA cache; their difference is compute, not storage.">
              <div className="stack" style={{ gap: 9 }}>
                {members.map((x) => {
                  const at1m = x.at["1048576"] ?? 0;
                  const hi = Math.max(...members.map((y) => y.at["1048576"] ?? 0));
                  return (
                    <div key={x.id} style={{ display: "grid", gridTemplateColumns: "76px 1fr 66px", gap: 8, alignItems: "center" }}>
                      <span className="mono" style={{ fontSize: 9.5 }}>{x.short}</span>
                      <div style={{ height: 11, background: "rgba(16,18,23,0.04)" }}>
                        <div style={{ height: "100%", width: `${Math.sqrt(at1m / hi) * 100}%`, background: x.scheme === "MLA" ? "var(--signal)" : "var(--ember)" }} />
                      </div>
                      <span className="mono" style={{ fontSize: 9, textAlign: "right" }}>{bytes(at1m)}</span>
                    </div>
                  );
                })}
              </div>
              <div className="caption">Bars are square-root scaled so V4.1 remains visible beside V3.</div>
            </Panel>
            <div className="grid2">
              <Stat k="V3 at 1M" v={bytes((members.find((x) => x.id === "v3")?.at["1048576"] ?? 0))} s="MLA, fp16" />
              <Stat k="V4.1 at 1M" v={bytes((members.find((x) => x.id === "v41flash")?.officialPerToken ?? 0) * 1048576)} s="official, FP4" />
            </div>
          </div>
        </div>

        <h2 className="sec">Inside V4's cache</h2>
        <Panel
          title={<><b>Figure 4.4</b> · per-layer compression and KV sharing</>}
          tools={<Legend items={[{ c: "#2b46f0", label: "4:1" }, { c: "#15246e", label: "128:1" }, { c: "#d9d5c8", label: "sliding" }]} />}
          note="V4 alternates 4:1 and 128:1 compressed layers; V4.1 stores only four source layers and lets the rest read them. The ring marks a layer that owns a cache. A missing ratio means a sliding-window layer, whose cache is bounded by 128 tokens."
        >
          <div className="row" style={{ marginBottom: 8 }}>
            <select className="ctl" value={sel} onChange={(e) => setSel(e.target.value)} style={{ maxWidth: 180 }}>
              {members.filter((x) => x.compress).map((x) => <option key={x.id} value={x.id}>{x.short}</option>)}
            </select>
          </div>
          {m && m.compress ? <CompressStrip m={m} /> : <div className="caption">This member uses MLA; its cache is uniform across layers.</div>}
          {m && m.compress ? (
            <div className="caption" style={{ marginTop: 6 }}>
              The strip is wider than a narrow column: scroll it horizontally to follow layers 20–39.
            </div>
          ) : null}
          {m && (
            <div className="grid4" style={{ marginTop: 10 }}>
              <Stat k="Scheme" v={m.scheme} s={m.compress ? "hybrid attention" : "latent KV"} />
              <Stat k="Stored layers" v={m.layersStored} s={`of ${m.layers}`} />
              <Stat k="Bytes / token" v={m.officialPerToken ? bytes(m.officialPerToken) : bytes(m.perToken)} s={m.officialPerToken ? "official, FP4" : "fp16, modelled"} />
              <Stat k="vs MHA" v={`${fmt(m.ratio, 0)}×`} s="fp16 model" />
            </div>
          )}
          {m?.officialPerToken ? (
            <div className="caption" style={{ borderTop: "1px solid var(--rule)", marginTop: 12, paddingTop: 9 }}>
              <strong>Official figure for V4.1-Flash: {m.officialPerToken} B/token</strong> at FP4 with a
              128-token sliding window — {bytes(m.officialPerToken * 1048576)} for a one-million-token
              sequence. The FP16 layer-plan model for the same checkpoint gives {bytes(m.perToken)}/token,
              because it does not model FP4 storage or the sliding-window bound. Both are shown; they
              answer different questions.
            </div>
          ) : null}
        </Panel>

        <Prose>
          <p>
            MLA's trick is not that it approximates attention. It is that the{" "}
            <strong>cache is a different tensor from the one attention reads</strong>.
            A token writes {mla?.kvLora ?? 512} latent channels plus a 64-channel
            RoPE key; each of the heads reconstructs its own K and V from that latent at
            read time. The model pays a small up-projection on every step and saves an
            order of magnitude in memory traffic. At batch 1, decode is bandwidth-bound,
            so that trade is decisive.
          </p>
          <p>
            V4 changes the axis of compression: instead of compressing <em>heads</em>,
            it compresses <em>sequence</em>. A 4:1 layer keeps one key per four tokens;
            a 128:1 layer keeps one per 128. The layers alternate so that local detail
            and global reach are both available. V4.1 goes further: four of the forty
            layers compute a full input cache (three in the encoder, one in the decoder),
            the rest read it, and every layer keeps a 128-token sliding window. That is
            why its line in Figure 4.2 is so flat — and why
            the published FP4 figure is 890 bytes per token, not the 2.8 KiB that this
            chapter's FP16 model computes for the same layer plan.
          </p>
        </Prose>

        <Panel
          title={<><b>Figure 4.5</b> · the lab's own claims</>}
          tight
          note="Kept separate from the computed figures: these are published claims, not measurements reproduced here."
        >
          <table className="tbl">
            <thead><tr><th>model</th><th>claim</th><th>source</th></tr></thead>
            <tbody>
              {(data?.official ?? []).map((o, i) => (
                <tr key={i}>
                  <td className="k">{o.model}</td>
                  <td style={{ fontFamily: "var(--sans)", fontSize: 11.5 }}>{o.claim}</td>
                  <td><a href={o.url} target="_blank" rel="noreferrer">link</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Method>
          <p>
            For MLA the cache is <code>L × (kv_lora_rank + qk_rope_head_dim) × bytes</code>{" "}
            per token. For V4 the cache is modelled per layer as{" "}
            <code>(head_dim + rope) / compress_ratio</code>, counting only layers that
            own a cache; layers with ratio zero use a 128-token sliding window and are
            reported separately. All figures assume fp16 and one sequence.
          </p>
          <p>
            The MHA reference uses the same number of query heads and head dimension as
            the checkpoint, so the ratio answers “what would this model cost without
            MLA?”, not “what did DeepSeek 67B cost?”. V4's ratios look extreme because
            its MHA reference has 128 heads of width 512.
          </p>
        </Method>

        <Footnotes
          items={[
            "Cache sizes scale linearly with batch size and with the number of concurrent sequences. No batching is assumed here.",
            "V4's per-token cache is an explicit FP16 model: the config names the compression ratios and KV source layers, but not the byte layout of the shared cache. Treat the absolute number as an estimate and the ordering as robust.",
            "V4.1's official 890 B/token is FP4 with a 128-token sliding window; the 2.8 KiB in the chart is the FP16 model of the same layer plan. They are different precision regimes and must not be compared directly.",
            "Quantising the cache to FP8 would halve every FP16 number in this chapter. The figures stay in fp16 so that generations are comparable.",
            "The 437× claim from DeepSeek compares V4.1 with the first-generation model, not with V3. The computed ratios here use each checkpoint's own MHA width, so they are larger; the two numbers answer different questions.",
          ]}
        />
      </DataState>
      <PageFoot index={4} />
    </>
  );
}
