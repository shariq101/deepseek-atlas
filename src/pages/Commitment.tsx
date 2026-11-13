import React, { useMemo, useState } from "react";
import { LensEntry, Meta, Prompt, fmt, pct, tokenLabel, useMeasure, useOne } from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, Scatter } from "../components/charts";
import { SEQ, ramp } from "../lib/atlas";

const COLORS = ["var(--signal)", "var(--ember)", "var(--teal)", "var(--olive)", "var(--plum)", "var(--sand)", "var(--ink-3)", "var(--ink-2)"];

export default function Commitment() {
  const metaQ = useOne<Meta>("meta");
  const promptQ = useOne<Prompt[]>("prompts");
  const lensQ = useOne<Record<string, LensEntry>>("lens");
  const { ref, width } = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const meta = metaQ.data;
  const prompts = promptQ.data ?? [];
  const lens = lensQ.data;
  const L = meta?.layers ?? 27;

  const rows = useMemo(() => {
    if (!lens) return [];
    return prompts.map((p) => {
      const e = lens[p.id];
      if (!e) return null;
      const commit = e.targetRank.findIndex((r) => r === 0);
      return {
        id: p.id, group: p.group, commit: commit < 0 ? L : commit,
        never: commit < 0, final: e.targetProb[L] ?? 0, text: p.text,
        rank: e.targetRank,
      };
    }).filter(Boolean) as {
      id: string; group: string; commit: number; never: boolean; final: number; text: string; rank: number[];
    }[];
  }, [prompts, lens, L]);

  const groups = Array.from(new Set(rows.map((r) => r.group)));
  const color = (g: string) => COLORS[groups.indexOf(g) % COLORS.length];
  const committed = rows.filter((r) => !r.never);
  const median = committed.length
    ? [...committed].sort((a, b) => a.commit - b.commit)[Math.floor(committed.length / 2)].commit : 0;
  const early = rows.filter((r) => r.commit <= L * 0.5).length;

  return (
    <>
      <ChapterHead n="07" part="II" kicker="Commitment"
        title={<>When a prediction becomes <em>inevitable</em></>}
        standfirst={<>Every prompt has a depth at which its target stops being a hypothesis and becomes the lens argmax. Plotting that layer against the final probability separates three kinds of decision: early and confident, late and contested, and never.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>{rows.length} prompts</Chip><Chip tone="blue">{L} layers</Chip></>}
      />
      <DataState loading={metaQ.loading || promptQ.loading || lensQ.loading}
        error={metaQ.error ?? promptQ.error ?? lensQ.error} what="the lens ranks">
        <div className="grid4">
          <Stat k="Median commitment" v={`L${median}`} s={`of ${L} layers`} />
          <Stat k="Committed by half depth" v={`${early} / ${rows.length}`} s="L ≤ 13" />
          <Stat k="Never commit" v={rows.filter((r) => r.never).length} s="target never #1" />
          <Stat k="Prompts" v={rows.length} s={`${groups.length} computation types`} />
        </div>

        <Panel title={<><b>Figure 7.1</b> · commitment depth against final confidence</>}
          tools={<Legend items={groups.map((g) => ({ c: color(g), label: g }))} />}
          note="Each dot is one prompt. X is the first layer at which the target is the lens argmax (right edge = never). Y is the final probability of the first target token. Early commitment with low final confidence is the interesting quadrant: the lens found the answer before the model did.">
          <div ref={ref}>
            <Scatter width={Math.max(width, 380)} height={330}
              points={rows.map((r, i) => ({
                x: r.commit, y: r.final, color: color(r.group), r: hover === i ? 7 : 4.5,
                label: `${r.id} · ${r.group} · commit L${r.commit} · final ${pct(r.final, 1)}`,
              }))}
              xDomain={[0, L]} yDomain={[0, 1]}
              xTicks={[0, 7, 14, 20, L]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
              xLabel="commitment layer (L = never)" yLabel="final P(target)"
              yFormat={(v) => pct(v, 0)} highlight={hover} onHover={setHover}
            />
          </div>
          <div className="hoverbox">
            {hover != null && rows[hover]
              ? <><b>{rows[hover].id}</b> · {rows[hover].group} · commits at L{rows[hover].commit} · final {pct(rows[hover].final, 1)} · “{rows[hover].text.slice(0, 70)}”</>
              : "Hover a point for its prompt and measured values."}
          </div>
        </Panel>

        <Panel title={<><b>Figure 7.2</b> · the rank of the target, all prompts</>}
          note="Rows are prompts, columns layers. A cell is the rank of the target token (0 = argmax); darker means earlier in the distribution. The diagonal frontier is the variety of commitments: some answers are decided in the first third, others at the last block."
          tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>{rows.length} × {L + 1}</span>}>
          <div style={{ overflowX: "auto" }}>
            <Heat rows={rows.length} cols={L + 1} cellW={20} cellH={13} gap={1}
              value={(r, c) => 1 - Math.min(1, Math.log1p(rows[r].rank[c] ?? 0) / Math.log1p(200))}
              color={(v) => ramp(v, SEQ)}
              rowLabels={(r) => rows[r].id.slice(0, 12)}
              colLabels={(c) => (c % 5 === 0 ? String(c) : "")}
              formatValue={(v) => `${(v * 100).toFixed(0)}%`}
            />
          </div>
          <div className="caption">Darker = target nearer the top of the lens distribution. The target's rank often falls sharply at one layer, then plateaus.</div>
        </Panel>

        <div className="split">
          <Panel title={<><b>Figure 7.3</b> · the distribution of commitment</>}
            note="Counts per layer, coloured by computation type. Factual recall tends to commit late; induction and code often commit earlier, when the continuation is strongly constrained.">
            <svg width={Math.max(width, 360)} height={220} className="chart">
              {Array.from({ length: L + 1 }, (_, l) => {
                const here = rows.filter((r) => r.commit === l);
                if (!here.length) return null;
                const x = 20 + (l / (L + 1)) * (Math.min(width, 760) - 50);
                let y = 190;
                return (
                  <g key={l}>
                    {groups.map((g) => {
                      const n = here.filter((r) => r.group === g).length;
                      if (!n) return null;
                      const h = n * 16;
                      y -= h;
                      return <rect key={g} x={x} y={y} width={Math.min(width, 760) / (L + 1) - 3} height={h - 1} fill={color(g)} />;
                    })}
                    <text x={x + 4} y={205} className="ticklabel">{l}</text>
                  </g>
                );
              })}
              <line x1={20} x2={Math.min(width, 760) - 30} y1={190} y2={190} stroke="var(--ink)" />
            </svg>
          </Panel>
          <Prose>
            <p>
              Commitment is not the same as correctness, and it is not the same as
              confidence. A prompt can commit early because the continuation is
              over-determined by a repeated pattern, while a later block may still revise
              the probability. Another can remain outside the top 50 for most of the depth
              and still end above 80% once the last blocks resolve it.
            </p>
            <p>
              The scatter is therefore a map of <em>when the computation becomes
              legible</em>, not a map of difficulty. The clearest cases are the
              late-commit/high-confidence prompts: the answer exists only after the
              whole depth has run.
            </p>
          </Prose>
        </div>

        <Method>
          <p>
            Commitment is the smallest layer <code>l</code> such that the target token
            is the argmax of the lens distribution at <code>l</code>, computed over the
            full vocabulary. Prompts whose target never reaches argmax are plotted at{" "}
            <code>L</code> and marked “never”; they are excluded from the median.
          </p>
          <p>
            Final confidence is the first-target-token probability at the final layer,
            which is the model's actual softmax. No temperature or calibration is applied.
          </p>
        </Method>

        <Footnotes items={[
          "The corpus is 27 short prompts; commitment statistics are descriptive and not a benchmark. A single prompt can move the median by a layer.",
          "The stored rank array is 0-based (0 = argmax); the page displays ranks 1-based. Ties are broken by the tokenizer order; with NF4 weights, near-ties can flip.",
          "‘Never’ does not mean the model has no answer; it means the target string is not the lens argmax, often because the continuation begins with a different tokenisation.",
          "The full per-prompt rank arrays are in lens.json; this page is a pure function of them.",
        ]} />
      </DataState>
      <PageFoot index={7} />
    </>
  );
}
