import React, { useMemo, useState } from "react";
import {
  AttentionEntry, Meta, Prompt, fmt, pct, tokenLabel, useMeasure, useOne, SEQ, ramp,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart, Scatter } from "../components/charts";

export default function Attention() {
  const metaQ = useOne<Meta>("meta");
  const promptQ = useOne<Prompt[]>("prompts");
  const attnQ = useOne<Record<string, AttentionEntry>>("attention");
  const [pid, setPid] = useState<string | null>(null);
  const [layer, setLayer] = useState(12);
  const [head, setHead] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const prompts = promptQ.data ?? [];
  const attention = attnQ.data;
  const prompt = prompts.find((p) => p.id === pid) ?? prompts.find((p) => attention?.[p.id]) ?? prompts[0];
  const a = prompt && attention ? attention[prompt.id] : null;
  const H = meta?.heads ?? 16;
  const L = a ? a.heads.length : (meta?.layers ?? 27);
  const keys = a?.tokens ?? [];
  const T = keys.length;

  const roles = useMemo(() => {
    if (!a) return [];
    return a.stats.map((row, l) => {
      const prev = row.reduce((s, h) => s + h.prev, 0) / row.length;
      const sink = row.reduce((s, h) => s + h.sink, 0) / row.length;
      const ind = row.reduce((s, h) => s + h.ind, 0) / row.length;
      const ent = row.reduce((s, h) => s + h.ent, 0) / row.length;
      return { l, prev, sink, ind, ent };
    });
  }, [a]);

  const atLayer = a?.stats[layer] ?? [];
  const meanRow = (key: keyof typeof atLayer[number]) =>
    atLayer.reduce((s, h) => s + h[key], 0) / Math.max(atLayer.length, 1);

  return (
    <>
      <ChapterHead n="11" part="II" kicker="Attention"
        title={<>What the heads <em>look at</em></>}
        standfirst={<>DeepSeek's MLA keeps the same 16 query heads as a conventional transformer but stores a compressed latent instead of per-head keys and values. The attention maps are therefore both familiar and unusually cheap — and they still show sinks, previous-token heads and induction.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>16 heads · MLA</Chip><Chip>512 latent + 64 rope</Chip></>}
      />
      <DataState loading={metaQ.loading || promptQ.loading || attnQ.loading}
        error={metaQ.error ?? promptQ.error ?? attnQ.error} what="the attention capture">
        {prompt && a && (
          <>
            <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
              <label className="field">
                <span>Prompt</span>
                <select className="ctl" value={prompt.id} onChange={(e) => setPid(e.target.value)} style={{ minWidth: 340 }}>
                  {prompts.filter((p) => attention?.[p.id]).map((p) => (
                    <option key={p.id} value={p.id}>{p.id} — {p.text.slice(0, 56).replace(/\n/g, " ")}</option>
                  ))}
                </select>
              </label>
              <div style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>{H} heads × {L} layers</span>
            </div>

            <Panel title={<><b>Figure 11.1</b> · sixteen heads at the final query</>}
              tools={<span className="mono" style={{ fontSize: 9 }}>L{layer} · H{head}</span>}
              note="Rows are the 16 query heads; columns are key positions at the final query. The bright first column is the attention sink. Previous-token mass sits near the diagonal, and other structure is content for this head, not a universal role. The capture stores this final-query row; it does not store every query position for every prompt.">
              <div className="row" style={{ gap: 18, alignItems: "flex-start", flexWrap: "wrap" }}>
                <div style={{ overflowX: "auto" }}>
                  <Heat rows={H} cols={T} cellW={Math.max(14, Math.min(22, 520 / Math.max(T, 1)))} cellH={14} gap={0.5}
                    value={(h, k) => a.heads[layer]?.[h]?.[k] ?? 0}
                    color={(v) => ramp(v, SEQ)}
                    rowLabels={(h) => `H${h}`}
                    colLabels={(k) => (tokenLabel(keys[k] ?? "") || "∅").slice(0, 6)}
                    formatValue={(v) => pct(v, 0)} />
                </div>
                <div style={{ minWidth: 190 }}>
                  <div className="mini-label">layer</div>
                  <input type="range" min={0} max={L - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} />
                  <div className="mono" style={{ fontSize: 10, color: "var(--ember)", marginBottom: 8 }}>L{layer}</div>
                  <div className="mini-label">head</div>
                  <input type="range" min={0} max={H - 1} value={head} onChange={(e) => setHead(+e.target.value)} />
                  <div className="mono" style={{ fontSize: 10, color: "var(--ember)" }}>H{head}</div>
                  <div style={{ marginTop: 12 }}>
                    <Stat k="prev-token" v={pct(a.stats[layer]?.[head]?.prev ?? 0, 0)} s="mass on q−1" />
                    <Stat k="sink" v={pct(a.stats[layer]?.[head]?.sink ?? 0, 0)} s="mass on position 0" />
                    <Stat k="induction" v={pct(a.stats[layer]?.[head]?.ind ?? 0, 0)} s="repeat match" />
                    <Stat k="entropy" v={fmt(a.stats[layer]?.[head]?.ent ?? 0, 2)} s="nats" />
                  </div>
                </div>
              </div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 11.2</b> · head roles at this layer</>}
                note="Each point is a head. X is previous-token mass, Y is induction score; colour is attention entropy (dark = focused). Heads cluster into roles rather than filling the space.">
                <div ref={ref}>
                  <Scatter width={Math.max(width / 2 - 20, 320)} height={250}
                    points={atLayer.map((h, i) => ({
                      x: h.prev, y: h.ind, r: hover === i ? 8 : 5,
                      color: `hsl(214 48% ${62 - Math.min(40, h.ent * 18)}%)`,
                      label: `H${i} · prev ${pct(h.prev, 0)} · sink ${pct(h.sink, 0)} · induction ${pct(h.ind, 0)} · entropy ${fmt(h.ent, 2)}`,
                    }))}
                    xLabel="previous-token mass" yLabel="induction score"
                    xFormat={(v) => pct(v, 0)} yFormat={(v) => pct(v, 0)}
                    highlight={hover} onHover={setHover} />
                </div>
              </Panel>

              <Panel title={<><b>Figure 11.3</b> · attention through depth</>}
                note="Layer means over heads for the selected prompt. The sink dominates almost everywhere. Previous-token mass varies by prompt, and the induction score has a clear middle-layer peak only for the induction prompt; read the selected prompt rather than a family-wide law.">
                <div ref={ref}>
                  <LineChart width={Math.max(width / 2 - 20, 320)} height={250}
                    series={[
                      { values: roles.map((r) => r.prev), color: "var(--ember)", label: "prev-token", width: 1.7 },
                      { values: roles.map((r) => r.ind), color: "var(--signal)", label: "induction", width: 1.7 },
                      { values: roles.map((r) => r.sink), color: "var(--teal)", label: "sink", width: 1.4 },
                    ]}
                    xTicks={[0, 7, 14, 20, L - 1]} xLabel="layer" yFormat={(v) => pct(v, 0)} />
                </div>
                <div className="legend" style={{ marginTop: 6 }}>
                  <Legend items={[{ c: "var(--ember)", label: "prev-token", line: true }, { c: "var(--signal)", label: "induction", line: true }, { c: "var(--teal)", label: "sink", line: true }]} />
                </div>
              </Panel>
            </div>

            <div className="grid4">
              <Stat k="Mean prev-token" v={pct(meanRow("prev"), 0)} s={`at L${layer}`} />
              <Stat k="Mean sink" v={pct(meanRow("sink"), 0)} s="position 0" />
              <Stat k="Mean induction" v={pct(meanRow("ind"), 0)} s="repeated-token match" />
              <Stat k="Mean entropy" v={fmt(meanRow("ent"), 2)} s="nats" />
            </div>

            <Prose>
              <p>
                The sink is the most robust object in the capture. Most heads put a large
                fraction of their mass on the first token, including heads whose content is
                clearly elsewhere. That is not a semantic decision; it is a no-op the
                softmax can use when a head has nothing to add. It also means “attention to
                position 0” should be removed before interpreting a map.
              </p>
              <p>
                The previous-token band and the induction score are the two roles that
                survive that correction. The former is a copy of the current token's
                predecessor, useful for local syntax; the latter looks for an earlier
                occurrence of the current token and copies what followed it. In the
                induction prompt, induction mass peaks around the middle layers; in the
                factual prompts it is near zero, so this is a prompt-level observation,
                not a universal depth profile.
              </p>
            </Prose>

            <Method>
              <p>
                Attention probabilities are averaged over the 16 query heads only where
                noted; Figure 11.1 shows all heads at the final query. <code>prev</code> is the
                mean mass on position <code>q−1</code>, <code>sink</code> on position 0,
                and <code>ind</code> is the mass on the position following an earlier
                occurrence of the current token.
              </p>
              <p>
                MLA computes attention from a 512-dimensional latent plus a 64-dimensional
                decoupled rope key. The maps shown are post-softmax and already include
                the absorbed projection; they are not a re-derivation from the latent.
              </p>
            </Method>

            <Footnotes items={[
              `Captured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); ${meta?.corpus ?? 27} prompts, final-query maps for ${Object.keys(attention ?? {}).length}.`,
              "The final query can read every earlier key; the map is not a full query-position matrix.",
              "Induction is a string-match proxy, not a causal test. Chapter 13 ablates heads and blocks to test whether the pattern matters.",
              "Attention weights are not explanations: a head can attend to a token and contribute nothing to the output.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={11} />
    </>
  );
}
