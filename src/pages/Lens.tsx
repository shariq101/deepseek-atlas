import React, { useEffect, useMemo, useState } from "react";
import {
  LensEntry, Meta, Prompt, bare, fmt, pct, tokenLabel, useMeasure, useOne,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Seg, Stat,
} from "../components/ui";
import { LineChart, Series } from "../components/charts";

const hit = (e: LensEntry, t: string) =>
  (e.targetTokens ?? []).includes(t) || bare(t) === bare(e.targetStr);

function Spectrum({ e, layer }: { e: LensEntry; layer: number }) {
  const items = e.top[layer]?.slice(0, 12) ?? [];
  const hi = Math.max(...items.map((d) => d[1]), 1e-4);
  return (
    <div className="stack" style={{ gap: 5 }}>
      {items.map(([t, p], i) => (
        <div key={i} style={{ display: "grid", gridTemplateColumns: "112px 1fr 58px", gap: 9, alignItems: "center" }}>
          <span className="mono" style={{
            fontSize: 11.5, textAlign: "right", whiteSpace: "nowrap", overflow: "hidden",
            textOverflow: "ellipsis", color: hit(e, t) ? "var(--ember)" : i === 0 ? "var(--ink)" : "var(--ink-2)",
          }} title={t}>{tokenLabel(t) || "∅"}</span>
          <div style={{ height: 13, background: "rgba(16,18,23,0.04)" }}>
            <div style={{
              width: `${(p / hi) * 100}%`, height: "100%",
              background: hit(e, t) ? "var(--ember)" : i === 0 ? "var(--signal)" : "var(--rule-2)",
            }} />
          </div>
          <span className="mono" style={{ fontSize: 9.5, color: hit(e, t) ? "var(--ember)" : "var(--ink-3)" }}>{pct(p, p > 0.1 ? 1 : 2)}</span>
        </div>
      ))}
      <div className="caption">{layer === e.top.length - 1 ? "The model's actual output distribution." : "A projection through the final norm and unembedding."}</div>
    </div>
  );
}

function River({ e, layer, onPick }: { e: LensEntry; layer: number; onPick: (l: number) => void }) {
  if (!e.river) return <div className="caption">This prompt was not captured with per-position rows. Choose a deep-dive prompt.</div>;
  const L = e.river.length - 1, T = e.river[0].pt.length;
  const cw = 29, ch = 18, left = 100, top = 22;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={left + (L + 1) * cw + 8} height={top + T * ch + 8}>
        {Array.from({ length: L + 1 }, (_, l) => (
          <text key={l} x={left + l * cw + cw / 2} y={11} textAnchor="middle" className="ticklabel">{l}</text>
        ))}
        {Array.from({ length: T }, (_, t) => (
          <g key={t}>
            <text x={left - 8} y={top + t * ch + 12} textAnchor="end" className="ticklabel">
              {(tokenLabel(e.tokens[t] ?? "") || "∅").slice(0, 9)}
            </text>
            {Array.from({ length: L + 1 }, (_, l) => {
              const r = e.river![l], p = r.pt[t] ?? 0, tok = r.top1[t] ?? "";
              return (
                <g key={l}>
                  <rect x={left + l * cw} y={top + t * ch} width={cw - 1.5} height={ch - 1.5}
                    fill={`rgba(43,70,240,${0.08 + Math.sqrt(p) * 0.84})`}
                    stroke={hit(e, tok) ? "var(--ember)" : "none"} strokeWidth={hit(e, tok) ? 1.2 : 0}
                    style={{ cursor: "pointer" }} onClick={() => onPick(l)}>
                    <title>{`L${l} · pos ${t} · ${tokenLabel(tok) || "∅"} · P(target) ${pct(p, 1)}`}</title>
                  </rect>
                  <text x={left + l * cw + cw / 2} y={top + t * ch + 12} textAnchor="middle"
                    style={{ fontFamily: "var(--mono)", fontSize: 7.4, fill: p > 0.55 ? "#fbf9f3" : "var(--ink-2)", pointerEvents: "none" }}>
                    {(tokenLabel(tok) || "∅").slice(0, 5)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
        <line x1={left + layer * cw + cw / 2} x2={left + layer * cw + cw / 2} y1={top - 4} y2={top + T * ch}
          stroke="var(--ember)" strokeWidth={1} opacity={0.5} />
      </svg>
      <div className="caption">Fill = P(target) at that position and depth; amber outline = lens top-1 is the target.</div>
    </div>
  );
}

export default function Lens() {
  const metaQ = useOne<Meta>("meta");
  const promptQ = useOne<Prompt[]>("prompts");
  const lensQ = useOne<Record<string, LensEntry>>("lens");
  const [pid, setPid] = useState<string | null>(null);
  const [layer, setLayer] = useState(14);
  const [view, setView] = useState<"spectrum" | "river">("spectrum");
  const { ref, width } = useMeasure<HTMLDivElement>();

  const meta = metaQ.data;
  const prompts = promptQ.data ?? [];
  const lens = lensQ.data;

  const demo = useMemo(() => {
    if (!prompts.length || !lens) return null;
    let best = prompts[0], gain = -Infinity;
    for (const p of prompts) {
      const e = lens[p.id];
      if (!e || p.n < 5) continue;
      const g = e.targetProb[e.targetProb.length - 1] - Math.max(...e.targetProb.slice(0, 3));
      if (g > gain) { gain = g; best = p; }
    }
    return best;
  }, [prompts, lens]);

  useEffect(() => { if (demo && !pid) setPid(demo.id); }, [demo, pid]);
  const prompt = prompts.find((p) => p.id === pid) ?? demo;
  const e = prompt && lens ? lens[prompt.id] : null;
  const L = meta?.layers ?? (e ? e.targetProb.length - 1 : 27);
  const commit = e ? e.targetRank.findIndex((r) => r === 0) : -1;
  const first = e ? e.targetProb.findIndex((v) => v > 0.001) : -1;
  const series: Series[] = e ? [
    { values: e.targetProb, color: "var(--ember)", label: "P(target)", width: 2 },
    { values: e.top1Prob, color: "var(--signal)", label: "P(lens top-1)", width: 1.4 },
  ] : [];

  return (
    <>
      <ChapterHead n="06" part="II" kicker="Logit lens"
        title={<>The answer, forming in <em>layers</em></>}
        standfirst={<>Apply the model's own final norm and unembedding to every intermediate state and you get a guess at the next token from every depth. The guesses are not the output, but their sequence is a time-lapse of the answer being decided.</>}
        chips={<><Chip tone="teal">measured · V2-Lite</Chip><Chip>{L} layers</Chip><Chip tone="blue">{meta?.precision ?? "nf4"}</Chip></>}
      />
      <DataState loading={metaQ.loading || promptQ.loading || lensQ.loading}
        error={metaQ.error ?? promptQ.error ?? lensQ.error} what="the lens capture">
        {prompt && e && (
          <>
            <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
              <label className="field">
                <span>Contrast prompt</span>
                <select className="ctl" value={prompt.id} onChange={(ev) => setPid(ev.target.value)} style={{ minWidth: 340 }}>
                  {prompts.map((p) => <option key={p.id} value={p.id}>{p.id} — {p.text.slice(0, 56).replace(/\n/g, " ")}</option>)}
                </select>
              </label>
              <div style={{ flex: 1 }} />
              <Seg value={view} options={[{ v: "spectrum", label: "spectrum" }, { v: "river", label: "prediction river" }]} onChange={setView} />
            </div>

            <Panel
              title={<><b>Figure 6.1</b> · {view === "spectrum" ? "the distribution at one depth" : "every position, every layer"}</>}
              tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>layer {layer} / {L}</span>}
              note={view === "spectrum"
                ? "The lens top-12 at the chosen depth, with true vocabulary probabilities. The target continuation is amber. It usually appears as a tail hypothesis before it wins."
                : "Rows are positions, columns layers. Fill is the target probability; the cell label is the lens argmax."}>
              {view === "spectrum" ? (
                <div className="grid2" style={{ gap: 28, alignItems: "start" }}>
                  <Spectrum e={e} layer={Math.min(layer, e.top.length - 1)} />
                  <div>
                    <div className="spread" style={{ marginBottom: 5 }}>
                      <span className="mini-label">depth scrubber</span>
                      <span className="mono" style={{ fontSize: 10, color: "var(--ember)" }}>L{layer}</span>
                    </div>
                    <input type="range" min={0} max={L} value={layer} onChange={(ev) => setLayer(+ev.target.value)} />
                    <div className="grid2" style={{ gap: 10, marginTop: 10 }}>
                      <Stat k="P(target)" v={pct(e.targetProb[layer] ?? 0, 2)} />
                      <Stat k="target rank" v={e.targetRank[layer] === 0 ? "#1" : `#${(e.targetRank[layer] ?? 0) + 1}`} />
                      <Stat k="entropy" v={fmt(e.entropy[layer] ?? 0, 2)} s="nats" />
                      <Stat k="KL → final" v={fmt(e.kl[layer] ?? 0, 3)} s="nats" />
                    </div>
                    <div style={{ marginTop: 16 }}>
                      <div className="caption" style={{ marginBottom: 5 }}>Leading tokens at this depth.</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                        {(e.top[layer] ?? []).slice(0, 10).map(([t, p], i) => (
                          <span key={i} className="token" title={pct(p, 2)}
                            style={{ borderColor: hit(e, t) ? "var(--ember)" : "var(--rule)", background: `rgba(43,70,240,${0.07 + p * 0.38})` }}>
                            {tokenLabel(t) || "∅"}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : <River e={e} layer={layer} onPick={setLayer} />}
            </Panel>

            <h2 className="sec">When does the answer take shape?</h2>
            <Panel title={<><b>Figure 6.2</b> · probability of the continuation, by depth</>}
              note="Hover to move the depth cursor. Before commitment the top-1 curve is another token; when the target becomes the lens argmax, the two curves meet rather than cross.">
              <div ref={ref}>
                <LineChart width={Math.max(width, 360)} height={280} series={series}
                  yDomain={[0, Math.max(0.05, ...e.targetProb, ...e.top1Prob) * 1.08]}
                  xTicks={[0, 4, 9, 14, 19, L]} xLabel="layer" yLabel="probability"
                  yFormat={(v) => pct(v, 0)} hoverIndex={layer} onHover={(i) => i != null && setLayer(i)} />
              </div>
              <div className="legend" style={{ marginTop: 8 }}>
                <Legend items={[{ c: "var(--ember)", label: "P(target)", line: true }, { c: "var(--signal)", label: "P(lens top-1)", line: true }]} />
              </div>
            </Panel>

            <div className="grid4">
              <Stat k="Final P(target)" v={pct(e.targetProb[L] ?? 0, 1)} s="first target token" />
              <Stat k="Final top-1" v={tokenLabel(e.finalTop1) || "—"} s={pct(e.top1Prob[L] ?? 0, 1)} />
              <Stat k="Commitment" v={commit >= 0 ? `L${commit}` : "never"} s="first layer at argmax" />
              <Stat k="First signal" v={first >= 0 ? `L${first}` : "—"} s="P(target) > 0.1%" />
            </div>

            {e.targetTokenProbs && e.targetTokenProbs.length > 1 && (
              <Panel title={<><b>Figure 6.3</b> · the continuation, token by token</>}
                note="The model's own probability of each subsequent token given the previous ones. The product is the whole-continuation probability.">
                <div className="row" style={{ alignItems: "center", gap: 10 }}>
                  {(e.targetTokens ?? []).map((t, i) => (
                    <div key={i} style={{ textAlign: "center" }}>
                      <div className="token" style={{ fontSize: 15, border: "1px solid var(--rule)" }}>{tokenLabel(t) || "∅"}</div>
                      <div className="mono" style={{ fontSize: 9, color: "var(--ink-3)", marginTop: 3 }}>{pct(e.targetTokenProbs?.[i] ?? 0, 1)}</div>
                    </div>
                  ))}
                  <div style={{ marginLeft: "auto" }} className="stat">
                    <div className="k">whole continuation</div>
                    <div className="v">{pct(e.targetSeqProb ?? 0, 3)}</div>
                  </div>
                </div>
              </Panel>
            )}

            <Prose>
              <p>
                The curve is rarely a smooth ramp. For this prompt the target sits in the noise
                floor until <strong>L{Math.max(first, 0)}</strong>, then{" "}
                {commit >= 0 ? <>becomes the leading hypothesis at <strong>L{commit}</strong> and stays there.</> :
                  <>never becomes the leading hypothesis; the model settles on a different continuation.</>}
                {" "}It ends at <strong>{pct(e.targetProb[L] ?? 0, 1)}</strong> against a lens top-1 probability of{" "}
                <strong>{pct(e.top1Prob[L] ?? 0, 1)}</strong>.
              </p>
              <p>
                The interesting region is before commitment. The target does not need to
                be near the top of the distribution to be readable: in this corpus it can
                sit anywhere from the single digits to thousands before the rank collapses
                at commitment. That is the signature of a model that has found the relevant
                fact but not resolved it against alternatives.
              </p>
            </Prose>

            <Method>
              <p>
                Every layer's residual at the final prompt position is passed through the final
                RMSNorm and unembedding. Layer <code>L</code> is the real output; earlier layers are
                projections. The target is the first token of the continuation, checked to be a
                prefix-consistent next token.
              </p>
              <p>
                Entropy and KL use the full 102,400-token vocabulary, not the top-12 shown. The lens
                is not calibrated: early high probability is a projection, not confidence.
              </p>
            </Method>

            <Footnotes items={[
              `Measured on ${meta?.modelName ?? "DeepSeek-V2-Lite"} in ${meta?.precision ?? "NF4"}; ${meta?.corpus ?? 27} prompts, ${meta?.tokens ?? 313} tokens. Quantisation perturbs probabilities; layer ordering is the robust part.`,
              "Layer 0 is the embedding state before any block. The final layer is already post-norm, so it is passed directly to the unembedding.",
              "The river is captured only for deep-dive prompts: 28 × T × 3 values per prompt (top-1 token, its probability, target probability).",
              "A lens top-1 is not a causal explanation; chapter 13 ablates the same blocks.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={6} />
    </>
  );
}
