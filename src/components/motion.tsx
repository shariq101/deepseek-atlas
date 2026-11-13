import React, { useEffect, useState } from "react";
import {
  AttentionEntry, DlaEntry, LensEntry, Prompt, RouterEntry,
  clamp, fmt, pct, tokenLabel, useOne,
} from "../lib/atlas";

export type TaskStep = {
  label: string;
  detail: string;
  gain: string;
};

function useLoop(length: number, ms: number, playing: boolean) {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    if (!playing || length < 2) return;
    const timer = window.setInterval(() => setIndex((i) => (i + 1) % length), ms);
    return () => window.clearInterval(timer);
  }, [length, ms, playing]);
  return [index, setIndex] as const;
}

export function TaskAnimation({
  title, steps, accent = "var(--signal)",
}: {
  title: string; steps: TaskStep[]; accent?: string;
}) {
  const [playing, setPlaying] = useState(true);
  const [index, setIndex] = useLoop(steps.length, 1550, playing);
  const step = steps[index];

  return (
    <div className="task-anim" style={{ "--accent": accent } as React.CSSProperties}>
      <div className="task-head">
        <div className="task-heading">
          <span className="mini-label">ANIMATED TASK</span>
          <b>{title}</b>
        </div>
        <div className="task-actions">
          <button type="button" className="task-btn" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="task-btn" onClick={() => { setIndex(0); setPlaying(true); }}>
            Replay
          </button>
        </div>
      </div>

      <div className="task-rail" style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}>
        {steps.map((s, i) => (
          <button
            type="button"
            key={s.label}
            className={"task-node" + (i === index ? " on" : "") + (i < index ? " done" : "")}
            onClick={() => { setIndex(i); setPlaying(false); }}
            aria-label={`Show ${s.label}`}
          >
            <span className="task-num">{String(i + 1).padStart(2, "0")}</span>
            <span className="task-label">{s.label}</span>
            <span className="task-detail">{s.detail}</span>
          </button>
        ))}
      </div>
      <div className="task-progress" aria-hidden="true">
        <span style={{ width: `${((index + 1) / steps.length) * 100}%` }} />
      </div>
      <div className="task-readout">
        <div>
          <span className="mini-label">NOW</span>
          <b>{step.label}</b>
          <p>{step.detail}</p>
        </div>
        <div className="task-gain">
          <span className="mini-label">WHAT IT ADDS</span>
          <p>{step.gain}</p>
        </div>
      </div>
    </div>
  );
}

const PLAYBACK_STAGES = [
  {
    short: "tokens",
    sub: "the input",
    title: "A string becomes ids",
    detail: "The sentence enters as tokens. At this point the model has ids, not meanings: 'capital' is still just a vocabulary address.",
    focus: [] as number[],
  },
  {
    short: "embed",
    sub: "+ position",
    title: "Vectors with position",
    detail: "Each id indexes an embedding table. A position signal is injected, so the same word can mean a different thing at a different place in the sequence.",
    focus: [] as number[],
  },
  {
    short: "blocks 1–9",
    sub: "local form",
    title: "Early blocks build local form",
    detail: "The early stack tends to resolve surface form: token identity, punctuation, nearby patterns. The labels at right are a teaching gloss, not measured features.",
    focus: [0, 2],
  },
  {
    short: "attention",
    sub: "bind the phrase",
    title: "Attention lets positions read each other",
    detail: "At the 'is' position, attention can read back to 'capital' and 'France'. This is the first place a relation can cross positions rather than live in one token.",
    focus: [1, 3, 4],
  },
  {
    short: "blocks 10–18",
    sub: "relation",
    title: "Middle blocks compose relations",
    detail: "The phrase 'capital of France' can be represented as a subject, while 'is' carries the predicate. The residual stream keeps the earlier token evidence available.",
    focus: [1, 3],
  },
  {
    short: "blocks 19–27",
    sub: "prediction",
    title: "Late blocks point at a continuation",
    detail: "Late writes move the final position toward a next-token state. The same object is still a residual vector; it has accumulated features rather than switched to a symbol.",
    focus: [4],
  },
  {
    short: "norm + unembed",
    sub: "logits",
    title: "The final vector becomes logits",
    detail: "A final RMSNorm stabilises the vector, and the unembedding matrix projects it to one score per vocabulary item. There is no word chosen inside the stack.",
    focus: [4],
  },
  {
    short: "softmax",
    sub: "a distribution",
    title: "Softmax makes the next-token bet",
    detail: "Softmax turns the logits into a distribution. The illustrative continuation is 'Paris'; a real model run would produce a measured distribution in chapter 11.",
    focus: [1, 3, 4],
  },
];

const PLAYBACK_TOKENS = [
  { word: "The", role: "determiner" },
  { word: "capital", role: "noun · subject" },
  { word: "of", role: "preposition" },
  { word: "France", role: "named entity" },
  { word: "is", role: "verb · copula" },
];

/* --------------------------------------------------------------------------
   Measured activation playback
   All numbers below come from public/data/{lens,attention,router,dla}.json,
   captured on V2-Lite (NF4) for 27 prompts; fact-capital is the default.
   Every prompt has a final-query per-head attention row; full T×T matrices
   are stored only for the four deep prompts. Nothing is sampled or synthesised.
   -------------------------------------------------------------------------- */

const PROMPT_GROUPS: { label: string; ids: string[] }[] = [
  { label: "Factual", ids: ["fact-capital", "fact-planet", "fact-author", "fact-element", "fact-river", "fact-currency"] },
  { label: "Negation", ids: ["neg-always", "neg-whale"] },
  { label: "Idiom", ids: ["idiom-bucket", "idiom-sword"] },
  { label: "Syntax", ids: ["syn-singular", "syn-plural", "syn-attract", "syn-quote"] },
  { label: "Induction", ids: ["ind-1", "ind-2", "ind-3"] },
  { label: "Arithmetic", ids: ["arith-add", "arith-mul", "arith-chain"] },
  { label: "Code", ids: ["code-add", "code-loop", "code-list"] },
  { label: "Multilingual", ids: ["multi-fr", "multi-de", "multi-es"] },
  { label: "Retrieval", ids: ["ret-code"] },
];

function ActivationEmpty({ children }: { children: React.ReactNode }) {
  return (
    <div className="actplay act-empty">
      <span className="mini-label">MEASURED ACTIVATION PLAYBACK</span>
      <p>{children}</p>
    </div>
  );
}

export function ActivationPlayback() {
  const lensState = useOne<Record<string, LensEntry>>("lens");
  const attentionState = useOne<Record<string, AttentionEntry>>("attention");
  const routerState = useOne<Record<string, RouterEntry>>("router");
  const dlaState = useOne<Record<string, DlaEntry>>("dla");
  const promptState = useOne<Prompt[]>( "prompts");
  const [promptId, setPromptId] = useState("fact-capital");

  const lens = lensState.data?.[promptId];
  const attention = attentionState.data?.[promptId] ?? null;
  const router = routerState.data?.[promptId];
  const dla = dlaState.data?.[promptId];
  const promptById = new Map((promptState.data ?? []).map((p) => [p.id, p]));
  const promptMeta = promptById.get(promptId);
  const attentionAvailable = Boolean(attention);
  const maxStep = lens ? Math.max(0, lens.targetProb.length - 1) : 0;

  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    setStep(0);
    setPlaying(true);
  }, [promptId]);

  useEffect(() => {
    if (!playing || maxStep < 1) return;
    const timer = window.setInterval(() => {
      setStep((s) => (s >= maxStep ? 0 : s + 1));
    }, 760);
    return () => window.clearInterval(timer);
  }, [playing, maxStep]);

  if (lensState.loading || attentionState.loading || routerState.loading || dlaState.loading) {
    return <ActivationEmpty>Loading the measured V2-Lite capture…</ActivationEmpty>;
  }
  if (lensState.error || routerState.error || dlaState.error) {
    return (
      <ActivationEmpty>
        The measured capture could not be loaded. Check that lens.json, attention.json,
        router.json and dla.json are present in public/data.
      </ActivationEmpty>
    );
  }
  if (!lens || !router || !dla) {
    return <ActivationEmpty>The selected prompt is not present in the capture.</ActivationEmpty>;
  }

  const readout = clamp(step, 0, maxStep);
  const targetProb = lens.targetProb[readout] ?? 0;
  const targetRank = lens.targetRank[readout] ?? 0;
  const top = lens.top[readout] ?? [];
  const maxTop = Math.max(0.0001, ...top.map(([, p]) => p));

  // Readout 0 is the embedding. Readout k > 0 follows block k. Every prompt has
  // a final-query attention row; the optional guard covers an incomplete artifact.
  const totalBlocks = attention?.shape[0] ?? 27;
  const block = readout === 0 ? 0 : Math.min(readout, totalBlocks);
  const attentionLayer = attention ? clamp(block - 1, 0, attention.shape[0] - 1) : 0;
  const routerIndex = clamp(block - 2, 0, router.top.length - 1);
  const routerLayer = router.layers?.[routerIndex] ?? routerIndex + 1;
  const routerRow = block >= 2
    ? [...(router.top[routerIndex]?.[4] ?? [])].sort((a, b) => b[1] - a[1])
    : [];
  const maxExpert = Math.max(0.0001, ...routerRow.map(([, w]) => w));
  const dlaIndex = clamp(block - 1, 0, dla.attn.length - 1);

  const keys = lens.tokens;
  const headsHere = attention ? attention.heads[attentionLayer] ?? [] : [];
  const maxHeat = Math.max(0.0001, ...headsHere.flatMap((h) => h ?? []));
  const meanRow = attention ? attention.mean[attentionLayer] ?? [] : [];
  const meanTop = meanRow
    .map((w, i) => ({ w, i }))
    .sort((a, b) => b.w - a.w)[0];

  const sparkW = 320;
  const sparkH = 62;
  const sparkPoints = lens.targetProb
    .map((p, i) => `${(i / Math.max(1, maxStep)) * sparkW},${sparkH - clamp(p, 0, 1) * sparkH}`)
    .join(" ");
  const sparkX = (readout / Math.max(1, maxStep)) * sparkW;
  const sparkY = sparkH - clamp(targetProb, 0, 1) * sparkH;

  const dlaMax = Math.max(
    0.001,
    ...dla.attn.map((v) => Math.abs(v)),
    ...dla.mlp.map((v) => Math.abs(v)),
  );
  const dlaW = 320;
  const dlaH = 56;
  const dlaX = (i: number) => (i / Math.max(1, dla.attn.length - 1)) * dlaW;
  const dlaY = (v: number) => dlaH / 2 - (v / dlaMax) * (dlaH / 2 - 5);
  const attnPoints = dla.attn.map((v, i) => `${dlaX(i)},${dlaY(v)}`).join(" ");
  const mlpPoints = dla.mlp.map((v, i) => `${dlaX(i)},${dlaY(v)}`).join(" ");

  return (
    <div className="actplay">
      <div className="actplay-head">
        <div className="actplay-title">
          <span className="mini-label">MEASURED ACTIVATION PLAYBACK · V2-Lite · NF4</span>
          <b>“{promptMeta?.text ?? lens.tokens.join("")}”</b>
          <small>{promptId} · target “{lens.targetStr.trim()}” · one forward pass · {maxStep + 1} lens readouts · {totalBlocks} blocks</small>
        </div>
        <div className="task-actions">
          <button type="button" className="task-btn" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="task-btn" onClick={() => { setStep(0); setPlaying(true); }}>
            Replay
          </button>
        </div>
      </div>

      <div className="actplay-prompt">
        <label>
          <span className="mini-label">PROMPT</span>
          <select className="ctl act-prompt-select" value={promptId} onChange={(e) => setPromptId(e.target.value)}>
            {PROMPT_GROUPS.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.ids.map((id) => {
                  const meta = promptById.get(id);
                  return <option key={id} value={id}>{id}{meta ? ` · ${meta.target.trim()}` : ""}</option>;
                })}
              </optgroup>
            ))}
          </select>
        </label>
        <div className="actplay-prompt-meta">
          <b>{promptMeta?.group ?? "captured prompt"}</b>
          <span>{attentionAvailable ? "attention map captured" : "attention not captured · lens, router and DLA still available"}</span>
        </div>
      </div>

      <div className="actplay-transport">
        <input
          className="act-slider"
          type="range"
          min={0}
          max={maxStep}
          value={readout}
          onChange={(e) => { setStep(Number(e.target.value)); setPlaying(false); }}
          aria-label="Lens readout layer"
        />
        <div className="actplay-transport-label">
          <b>{readout === 0 ? "embedding + position" : `after block ${readout}`}</b>
          <span>readout {readout} / {maxStep}</span>
        </div>
      </div>

      <div className="actplay-stages" aria-hidden="true">
        {lens.targetProb.map((_, i) => (
          <button
            type="button"
            key={i}
            className={"act-stage-tick" + (i === readout ? " on" : "") + (i < readout ? " done" : "")}
            onClick={() => { setStep(i); setPlaying(false); }}
            tabIndex={-1}
            aria-label={`Readout ${i}`}
          />
        ))}
      </div>

      <div className="actplay-grid">
        <section className="act-card act-lens">
          <header className="act-card-head">
            <div>
              <span className="mini-label">LOGIT LENS · TARGET READOUT</span>
              <b>What the residual already points at</b>
            </div>
            <div className="act-target-chip">
              <span>target</span>
              <b>{lens.targetStr.trim()}</b>
              <small>rank #{targetRank.toLocaleString()}</small>
            </div>
          </header>

          <div className="act-preds">
            {top.slice(0, 4).map(([token, p], i) => (
              <div className="act-pred" key={`${token}-${i}`}>
                <span className="act-pred-rank">{String(i + 1).padStart(2, "0")}</span>
                <span className="act-pred-token">{tokenLabel(token).trim() || token}</span>
                <span className="act-pred-track">
                  <i style={{ width: `${Math.max(1, (p / maxTop) * 100)}%` }} />
                </span>
                <b>{pct(p, p < 0.001 ? 3 : 1)}</b>
              </div>
            ))}
          </div>

          <div className="act-target-row">
            <span>target probability</span>
            <b>{pct(targetProb, targetProb < 0.001 ? 3 : 2)}</b>
            <span className="act-target-track">
              <i style={{ width: `${Math.max(1.5, clamp(targetProb, 0, 1) * 100)}%` }} />
            </span>
          </div>

          <div className="act-spark">
            <div className="act-spark-head">
              <span className="mini-label">TARGET PROBABILITY · ALL READOUTS</span>
              <span>final {pct(lens.targetProb[maxStep] ?? 0, 2)} · top-1 {lens.finalTop1.trim()}</span>
            </div>
            <svg viewBox={`0 0 ${sparkW} ${sparkH + 10}`} role="img" aria-label="Target probability by lens readout">
              <line x1={0} y1={sparkH} x2={sparkW} y2={sparkH} stroke="var(--rule-2)" strokeWidth={0.7} />
              <line x1={0} y1={sparkH * 0.5} x2={sparkW} y2={sparkH * 0.5} stroke="var(--rule)" strokeWidth={0.5} strokeDasharray="2 3" />
              <polyline points={sparkPoints} fill="none" stroke="var(--signal)" strokeWidth={1.6} />
              <line x1={sparkX} y1={0} x2={sparkX} y2={sparkH} stroke="var(--signal)" strokeWidth={0.8} opacity={0.45} />
              <circle cx={sparkX} cy={sparkY} r={3.2} fill="var(--signal)" />
              <text x={0} y={sparkH + 9} className="chart-endlabel">0</text>
              <text x={sparkW} y={sparkH + 9} textAnchor="end" className="chart-endlabel">{maxStep}</text>
            </svg>
          </div>
        </section>

        <section className="act-card act-attn">
          <header className="act-card-head">
            <div>
              <span className="mini-label">ATTENTION · FINAL QUERY “{lens.tokens[lens.tokens.length - 1]?.trim().toUpperCase()}”</span>
              <b>{attentionAvailable ? "Sixteen heads read the prefix" : "Head map not captured"}</b>
            </div>
            <span className={"act-layer-tag" + (attentionAvailable ? "" : " muted")}>{attentionAvailable ? (block === 0 ? "embedding" : `block ${block}`) : "not captured"}</span>
          </header>

          {!attentionAvailable ? (
            <div className="act-slot-empty">
              No attention row is present for this prompt in the capture.
              The lens, router and direct attribution are still measured.
            </div>
          ) : block === 0 ? (
            <div className="act-slot-empty">No block has run yet. Press play, or move the readout slider.</div>
          ) : (
            <>
              <div className="act-heat" style={{ gridTemplateColumns: `24px repeat(${keys.length}, minmax(0, 1fr))` }}>
                <span />
                {keys.map((k) => <span className="act-heat-key" key={k}>{k}</span>)}
                {headsHere.map((h, hi) => (
                  <React.Fragment key={hi}>
                    <span className="act-heat-head">{hi}</span>
                    {keys.map((_, ki) => {
                      const w = h[ki] ?? 0;
                      return (
                        <span
                          className="act-heat-cell"
                          key={ki}
                          title={`head ${hi} → ${keys[ki]}: ${fmt(w, 3)}`}
                          style={{ background: `rgba(43,70,240,${(0.035 + (w / maxHeat) * 0.9).toFixed(3)})` }}
                        />
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
              <div className="act-attn-foot">
                <span className="mini-label">MEAN ACROSS 16 HEADS</span>
                <div className="act-mean-bars">
                  {meanRow.map((w, i) => (
                    <span key={keys[i] ?? i} title={`${keys[i] ?? i}: ${fmt(w, 3)}`}>
                      <i style={{ height: `${Math.max(3, w * 100)}%` }} />
                    </span>
                  ))}
                </div>
                <p>
                  Top mean key: <b>{meanTop ? keys[meanTop.i] : "—"}</b> at {meanTop ? pct(meanTop.w) : "—"}.
                  A high weight is a routing fact, not a semantic explanation.
                </p>
              </div>
            </>
          )}
        </section>

        <section className="act-card act-router">
          <header className="act-card-head">
            <div>
              <span className="mini-label">ROUTER · FINAL TOKEN</span>
              <b>Which experts fire for “{lens.tokens[4]?.trim()}”</b>
            </div>
            <span className="act-layer-tag">{block >= 2 ? `layer ${routerLayer}` : "dense block"}</span>
          </header>
          {block < 2 ? (
            <div className="act-slot-empty">The first block has no routed-expert readout in this capture. Move forward to see the top-6 gate.</div>
          ) : (
            <div className="act-experts">
              {routerRow.map(([id, w]) => (
                <div className="act-expert" key={id}>
                  <span>expert {id}</span>
                  <i><b style={{ width: `${Math.max(2, (w / maxExpert) * 100)}%` }} /></i>
                  <em>{w.toFixed(3)}</em>
                </div>
              ))}
              <p className="act-card-note">These are the six selected gate weights. Expert identity is not a topic label.</p>
            </div>
          )}
        </section>

        <section className="act-card act-dla">
          <header className="act-card-head">
            <div>
              <span className="mini-label">DIRECT LOGIT ATTRIBUTION · TARGET</span>
              <b>Attention vs MLP contribution</b>
            </div>
            <span className="act-layer-tag">{block === 0 ? "embedding" : `block ${block}`}</span>
          </header>
          <div className="act-dla-stats">
            <div><span>attention</span><b style={{ color: "var(--signal)" }}>{block === 0 ? "—" : fmt(dla.attn[dlaIndex] ?? 0, 2)}</b></div>
            <div><span>MLP</span><b style={{ color: "var(--ember)" }}>{block === 0 ? "—" : fmt(dla.mlp[dlaIndex] ?? 0, 2)}</b></div>
          </div>
          <svg className="act-dla-chart" viewBox={`0 0 ${dlaW} ${dlaH}`} role="img" aria-label="Attention and MLP direct logit attribution by block">
            <line x1={0} y1={dlaH / 2} x2={dlaW} y2={dlaH / 2} stroke="var(--rule-2)" strokeWidth={0.7} />
            {block > 0 && <line x1={dlaX(dlaIndex)} y1={0} x2={dlaX(dlaIndex)} y2={dlaH} stroke="var(--ink-3)" strokeWidth={0.8} opacity={0.5} />}
            <polyline points={attnPoints} fill="none" stroke="var(--signal)" strokeWidth={1.2} opacity={0.85} />
            <polyline points={mlpPoints} fill="none" stroke="var(--ember)" strokeWidth={1.2} opacity={0.85} />
          </svg>
          <div className="act-legend">
            <span><i style={{ background: "var(--signal)" }} /> attention</span>
            <span><i style={{ background: "var(--ember)" }} /> MLP</span>
            <span>target logit</span>
          </div>
          <p className="act-card-note">A linear decomposition of the target logit. It is descriptive, not an intervention.</p>
        </section>
      </div>

      <div className="actplay-foot">
        <b>What this animation is.</b> The selected prompt's lens, router and attribution numbers are
        read directly from the V2-Lite capture; its attention panel is the final query's per-head row.
        Full T×T attention matrices are stored for the four deep prompts, while every prompt has this
        final-query row. The logit lens reads each intermediate residual through the final
        unembedding; it is not a decoder of meaning. Attention shows where the final query reads; it
        does not show what the head writes. The router shows which six experts ran; it does not name
        their function. The attribution is linear and descriptive, not causal.
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------
   Compact side-by-side comparison. The left side is fixed to fact-capital;
   the right side is one of the captured negation or idiom prompts. Both cards
   read the same lens index, so the comparison is at one shared readout.
   -------------------------------------------------------------------------- */

const COMPARISON_GROUPS = PROMPT_GROUPS.filter((g) => g.label === "Negation" || g.label === "Idiom");

function ComparisonCard({
  role, lens, attention, prompt, readout, maxStep,
}: {
  role: "baseline" | "contrast";
  lens: LensEntry;
  attention: AttentionEntry | null;
  prompt: Prompt | undefined;
  readout: number;
  maxStep: number;
}) {
  const targetProb = lens.targetProb[readout] ?? 0;
  const targetRank = lens.targetRank[readout] ?? 0;
  const top = lens.top[readout] ?? [];
  const topHere = top[0]?.[0] ?? "—";
  const maxTop = Math.max(0.0001, ...top.map(([, p]) => p));
  const keys = lens.tokens;
  const attentionLayer = readout === 0 ? -1 : clamp(readout - 1, 0, (attention?.shape[0] ?? 1) - 1);
  const headRows = attention && attentionLayer >= 0 ? attention.heads[attentionLayer] ?? [] : [];
  const maxHeat = Math.max(0.0001, ...headRows.flatMap((h) => h ?? []));
  const meanRow = attention && attentionLayer >= 0 ? attention.mean[attentionLayer] ?? [] : [];
  const meanTop = meanRow
    .map((w, i) => ({ w, i }))
    .sort((a, b) => b.w - a.w)[0];

  const sparkW = 250;
  const sparkH = 34;
  const sparkPoints = lens.targetProb
    .map((p, i) => `${(i / Math.max(1, maxStep)) * sparkW},${sparkH - clamp(p, 0, 1) * sparkH}`)
    .join(" ");
  const sparkX = (readout / Math.max(1, maxStep)) * sparkW;
  const sparkY = sparkH - clamp(targetProb, 0, 1) * sparkH;

  return (
    <section className={`actcmp-card ${role}`}>
      <div className="actcmp-card-head">
        <span className="mini-label">{role === "baseline" ? "BASELINE · FIXED" : "CONTRAST"}</span>
        <span className="actcmp-id">{prompt?.id ?? lens.targetStr.trim()}</span>
      </div>
      <b className="actcmp-prompt">“{prompt?.text ?? lens.tokens.join("")}”</b>
      <div className="actcmp-target-row">
        <span>target</span>
        <b>{lens.targetStr.trim()}</b>
        <span className="actcmp-rank">rank #{targetRank.toLocaleString()}</span>
      </div>

      <div className="actcmp-metrics">
        <div>
          <span>P(target)</span>
          <b>{pct(targetProb, targetProb < 0.001 ? 3 : 1)}</b>
        </div>
        <div>
          <span>final</span>
          <b>{pct(lens.targetProb[maxStep] ?? 0, 1)}</b>
        </div>
        <div>
          <span>top-1 here</span>
          <b>{tokenLabel(topHere).trim() || topHere}</b>
        </div>
      </div>

      <div className="actcmp-spark">
        <svg viewBox={`0 0 ${sparkW} ${sparkH + 6}`} role="img" aria-label={`${prompt?.id ?? "prompt"} target probability by readout`}>
          <line x1={0} y1={sparkH} x2={sparkW} y2={sparkH} stroke="var(--rule-2)" strokeWidth={0.7} />
          <line x1={0} y1={sparkH * 0.5} x2={sparkW} y2={sparkH * 0.5} stroke="var(--rule)" strokeWidth={0.5} strokeDasharray="2 3" />
          <polyline points={sparkPoints} fill="none" stroke={role === "baseline" ? "var(--signal)" : "var(--ember)"} strokeWidth={1.4} />
          <line x1={sparkX} y1={0} x2={sparkX} y2={sparkH} stroke={role === "baseline" ? "var(--signal)" : "var(--ember)"} strokeWidth={0.7} opacity={0.4} />
          <circle cx={sparkX} cy={sparkY} r={2.7} fill={role === "baseline" ? "var(--signal)" : "var(--ember)"} />
        </svg>
        <span className="actcmp-spark-label">all readouts · 0–{maxStep}</span>
      </div>

      <div className="actcmp-preds">
        {top.slice(0, 3).map(([token, p], i) => (
          <div className="actcmp-pred" key={`${token}-${i}`}>
            <span>{String(i + 1).padStart(2, "0")}</span>
            <b>{tokenLabel(token).trim() || token}</b>
            <i><em style={{ width: `${Math.max(2, (p / maxTop) * 100)}%` }} /></i>
            <small>{pct(p, p < 0.001 ? 3 : 1)}</small>
          </div>
        ))}
      </div>

      <div className="actcmp-attn">
        <div className="actcmp-attn-head">
          <span className="mini-label">ATTENTION · FINAL QUERY</span>
          <b className={attention ? "ok" : "off"}>{attention ? (readout === 0 ? "embedding" : "captured") : "not captured"}</b>
        </div>
        {!attention ? (
          <div className="actcmp-noattn">
            No attention row was captured for this prompt. The lens values remain measured.
          </div>
        ) : readout === 0 ? (
          <div className="actcmp-noattn">Embedding readout: no block has run yet. Move the slider to see the head map.</div>
        ) : (
          <>
            <div className="actcmp-heat" style={{ gridTemplateColumns: `repeat(${keys.length}, minmax(0, 1fr))` }}>
              {headRows.map((h, hi) => (
                <React.Fragment key={hi}>
                  {keys.map((_, ki) => {
                    const w = h[ki] ?? 0;
                    return (
                      <span
                        className="actcmp-heat-cell"
                        key={ki}
                        title={`head ${hi} → ${keys[ki]}: ${fmt(w, 3)}`}
                        style={{ background: `rgba(43,70,240,${(0.035 + (w / maxHeat) * 0.9).toFixed(3)})` }}
                      />
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
            <div className="actcmp-attn-foot">
              <span>16 heads · top mean key <b>{meanTop ? keys[meanTop.i] : "—"}</b>{meanTop ? ` at ${pct(meanTop.w)}` : ""}</span>
            </div>
          </>
        )}
      </div>

      <div className="actcmp-card-foot">
        {readout === 0 ? "embedding + position" : `after block ${readout}`} · readout {readout} / {maxStep}
      </div>
    </section>
  );
}

export function ActivationComparison() {
  const lensState = useOne<Record<string, LensEntry>>("lens");
  const attentionState = useOne<Record<string, AttentionEntry>>("attention");
  const promptState = useOne<Prompt[]>("prompts");
  const [comparisonId, setComparisonId] = useState("neg-always");
  const [step, setStep] = useState(27);
  const [playing, setPlaying] = useState(false);

  const baseLens = lensState.data?.["fact-capital"];
  const contrastLens = lensState.data?.[comparisonId];
  const promptById = new Map((promptState.data ?? []).map((p) => [p.id, p]));
  const maxStep = Math.max(0, Math.min(
    baseLens ? baseLens.targetProb.length - 1 : 27,
    contrastLens ? contrastLens.targetProb.length - 1 : 27,
  ));

  useEffect(() => {
    if (!playing || maxStep < 1) return;
    const timer = window.setInterval(() => {
      setStep((s) => (s >= maxStep ? 0 : s + 1));
    }, 900);
    return () => window.clearInterval(timer);
  }, [playing, maxStep]);

  if (lensState.loading || attentionState.loading || promptState.loading) {
    return <ActivationEmpty>Loading the side-by-side comparison…</ActivationEmpty>;
  }
  if (lensState.error || !baseLens || !contrastLens) {
    return <ActivationEmpty>The comparison capture is not available for this prompt pair.</ActivationEmpty>;
  }

  const readout = clamp(step, 0, maxStep);
  const baseProb = baseLens.targetProb[readout] ?? 0;
  const contrastProb = contrastLens.targetProb[readout] ?? 0;
  const deltaPts = (contrastProb - baseProb) * 100;
  const deltaLabel = Math.abs(deltaPts) < 0.05
    ? "0.0 pp"
    : `${deltaPts > 0 ? "+" : "−"}${Math.abs(deltaPts).toFixed(1)} pp`;
  const optionIds = COMPARISON_GROUPS.flatMap((g) => g.ids);

  return (
    <div className="actcmp">
      <div className="actcmp-head">
        <div>
          <span className="mini-label">MEASURED ACTIVATION PLAYBACK · SAME READOUT</span>
          <b>Fact-capital versus a contrast</b>
        </div>
        <div className="task-actions">
          <button
            type="button"
            className="task-btn"
            onClick={() => {
              if (!playing && readout >= maxStep) setStep(0);
              setPlaying((p) => !p);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="task-btn" onClick={() => { setStep(0); setPlaying(true); }}>
            Replay
          </button>
        </div>
      </div>

      <div className="actcmp-controls">
        <label className="actcmp-select-label">
          <span className="mini-label">CONTRAST PROMPT</span>
          <select className="ctl actcmp-select" value={comparisonId} onChange={(e) => setComparisonId(e.target.value)}>
            {COMPARISON_GROUPS.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.ids.map((id) => {
                  const meta = promptById.get(id);
                  return <option key={id} value={id}>{id}{meta ? ` · ${meta.target.trim()}` : ""}</option>;
                })}
              </optgroup>
            ))}
          </select>
        </label>
        <div className="actcmp-transport">
          <input
            className="act-slider"
            type="range"
            min={0}
            max={maxStep}
            value={readout}
            onChange={(e) => { setStep(Number(e.target.value)); setPlaying(false); }}
            aria-label="Shared lens readout for both prompts"
          />
          <span className="actcmp-readout">{readout === 0 ? "embedding" : `block ${readout}`} · {readout}/{maxStep}</span>
        </div>
      </div>

      <div className="actcmp-grid">
        <ComparisonCard
          role="baseline"
          lens={baseLens}
          attention={attentionState.data?.["fact-capital"] ?? null}
          prompt={promptById.get("fact-capital")}
          readout={readout}
          maxStep={maxStep}
        />
        <ComparisonCard
          role="contrast"
          lens={contrastLens}
          attention={attentionState.data?.[comparisonId] ?? null}
          prompt={promptById.get(comparisonId)}
          readout={readout}
          maxStep={maxStep}
        />
      </div>

      <div className="actcmp-delta">
        <span className="mini-label">RIGHT MINUS LEFT · TARGET PROBABILITY</span>
        <b>{deltaLabel}</b>
        <span>at readout {readout} · separate forward passes, same readout index</span>
      </div>
      <div className="actcmp-note">
        Left side is fixed to fact-capital. Right side is one of the four captured negation or idiom prompts.
        Attention is shown only where a full matrix exists; a missing matrix is reported, not imputed.
        There are {optionIds.length} contrast prompts.
      </div>
    </div>
  );
}

export function TransformerPlayback() {
  const [playing, setPlaying] = useState(true);
  const [stage, setStage] = useLoop(PLAYBACK_STAGES.length, 1900, playing);
  const current = PLAYBACK_STAGES[stage];
  const atEnd = stage >= PLAYBACK_STAGES.length - 1;

  return (
    <div className="xplay">
      <div className="xplay-head">
        <div>
          <span className="mini-label">PLAY THE FORWARD PASS</span>
          <b>The whole stack, one step at a time</b>
        </div>
        <div className="task-actions">
          <button type="button" className="task-btn" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="task-btn" onClick={() => { setStage(0); setPlaying(true); }}>
            Replay
          </button>
        </div>
      </div>

      <div className="xplay-progress" aria-hidden="true">
        <span style={{ width: `${((stage + 1) / PLAYBACK_STAGES.length) * 100}%` }} />
      </div>

      <div className="xplay-grid">
        <div className="xplay-panel">
          <span className="mini-label">SURFACE SEQUENCE</span>
          <div className="xplay-tokens">
            {PLAYBACK_TOKENS.map((t, i) => (
              <span
                key={t.word}
                className={"xplay-token" + (current.focus.includes(i) ? " focus" : "")}
              >
                <b>{t.word}</b>
                <small>{stage >= 2 ? t.role : "—"}</small>
              </span>
            ))}
          </div>
          <p className="xplay-caption">
            Teaching gloss: the roles are linguistic labels for this example, not a readout of the model's internal features.
          </p>
        </div>

        <div className="xplay-stack" aria-label="forward-pass stages">
          {PLAYBACK_STAGES.map((s, i) => (
            <button
              type="button"
              key={s.short}
              className={"xplay-stage" + (i === stage ? " on" : "") + (i < stage ? " done" : "")}
              onClick={() => { setStage(i); setPlaying(false); }}
            >
              <span>{s.short}</span>
              <small>{s.sub}</small>
            </button>
          ))}
        </div>

        <div className="xplay-monitor">
          <span className="mini-label">SEMANTIC MONITOR</span>
          <h3>{current.title}</h3>
          <p>{current.detail}</p>
          <div className="xplay-readout">
            <div>
              <span className="mini-label">CURRENT WRITE</span>
              <b>{current.short}</b>
            </div>
            <div>
              <span className="mini-label">ILLUSTRATIVE TOKENS</span>
              <b>{current.focus.length ? current.focus.map((i) => PLAYBACK_TOKENS[i].word).join(" · ") : "—"}</b>
            </div>
          </div>
          <div className={"xplay-semantics" + (stage >= 3 ? " on" : "")}>
            <div>
              <span className="mini-label">SUBJECT PHRASE · GLOSS</span>
              <b>capital of France</b>
            </div>
            <div>
              <span className="mini-label">PREDICATE · GLOSS</span>
              <b>is</b>
            </div>
          </div>
          <div className={"xplay-next" + (atEnd ? " on" : "")}>
            <span className="mini-label">NEXT TOKEN · ILLUSTRATIVE</span>
            <b>{atEnd ? "Paris" : "not yet"}</b>
            <small>{atEnd ? "Not a measured model prediction" : "The final distribution appears at the end"}</small>
          </div>
        </div>
      </div>

      <div className="xplay-timeline">
        {PLAYBACK_STAGES.map((s, i) => (
          <button
            type="button"
            key={s.short}
            className={"xplay-tick" + (i === stage ? " on" : "")}
            onClick={() => { setStage(i); setPlaying(false); }}
            aria-label={`Show ${s.short}`}
          >
            {String(i + 1).padStart(2, "0")}
          </button>
        ))}
        <span className="xplay-timeline-label">{current.title}</span>
      </div>
    </div>
  );
}
