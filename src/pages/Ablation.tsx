import React, { useMemo, useState } from "react";
import { Ablation, Meta, fmt, pct, useMeasure, useOne, SEQ, ramp } from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Seg, Stat,
} from "../components/ui";
import { Heat, LineChart } from "../components/charts";

type Kind = "block" | "attn" | "mlp";

export default function AblationPage() {
  const metaQ = useOne<Meta>("meta");
  const abQ = useOne<Ablation>("ablation");
  const [kind, setKind] = useState<Kind>("block");
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const ab = abQ.data;
  const L = ab?.layers.length ?? meta?.layers ?? 27;

  const topExperts = useMemo(() => {
    if (!ab) return [];
    return [...ab.experts].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 12);
  }, [ab]);

  const worst = useMemo(() => {
    if (!ab) return null;
    return [...ab.layers].sort((a, b) => b[kind].delta - a[kind].delta)[0];
  }, [ab, kind]);

  return (
    <>
      <ChapterHead n="13" part="II" kicker="Ablation"
        title={<>What happens when you <em>remove it</em></>}
        standfirst={<>Correlation is cheap. This chapter deletes one component at a time — a whole block, its attention, its MLP, a single expert, the shared expert — and measures the change in next-token negative log-likelihood. The result is a causal ranking, not a map of activity.</>}
        chips={<><Chip tone="teal">measured · causal</Chip><Chip>{ab?.prompts.length ?? 12} prompts</Chip><Chip>NLL Δ</Chip></>}
      />
      <DataState loading={metaQ.loading || abQ.loading} error={metaQ.error ?? abQ.error} what="the ablation sweep">
        {ab && (
          <>
            <div className="grid4">
              <Stat k="Baseline NLL" v={fmt(ab.baseline, 3)} s="mean over prompts" />
              <Stat k="Most damaging" v={worst ? `L${worst.l}` : "—"} s={worst ? `+${fmt(worst[kind].delta, 2)} nats` : ""} />
              <Stat k="Ablations" v={ab.layers.length * 3 + ab.experts.length + ab.shared.length} s="interventions, each over prompts" />
              <Stat k="Expert knockouts" v={ab.experts.length} s="one expert at a time" />
            </div>

            <Panel title={<><b>Figure 13.1</b> · knockout curve, by depth</>}
              tools={<Seg value={kind} options={[{ v: "block", label: "block" }, { v: "attn", label: "attention" }, { v: "mlp", label: "MLP" }]} onChange={setKind} />}
              note="Δ NLL when the attention or MLP output is set to zero, or the whole block is removed from the residual path. Positive means the prompt got worse; negative means the component was hurting. The shape, not the absolute value, is the finding.">
              <div ref={ref}>
                <LineChart width={Math.max(width, 380)} height={280}
                  series={[{ values: ab.layers.map((l) => l[kind].delta), color: kind === "mlp" ? "var(--ember)" : "var(--signal)", label: `Δ NLL · ${kind}`, width: 2 }]}
                  xTicks={[0, 7, 14, 20, L - 1]} xLabel="layer" yLabel="Δ NLL (nats)"
                  yFormat={(v) => (v > 0 ? "+" : "") + fmt(v, 2)} />
              </div>
              <div className="caption">Mean over {ab.prompts.length} prompts. A flat region is a component the model can lose without changing the answer; a peak is a necessary write.</div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 13.2</b> · the damage is not uniform across prompts</>}
                note="Rows are prompts, columns layers; colour is Δ NLL for the selected knockout, clipped to 0–2 nats and shown as a percentage of that ceiling. A vertical band means every prompt depends on that layer; a single dark cell means one prompt does. Hover for the underlying value.">
                <Heat rows={ab.prompts.length} cols={L} cellW={Math.max(9, 360 / L)} cellH={11} gap={0.8}
                  value={(p, l) => {
                    const v = ab.layers[l]?.[kind].perPrompt[p] ?? 0;
                    return Math.min(1, Math.max(0, v / 2));
                  }}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(p) => ab.prompts[p]?.slice(0, 12)}
                  colLabels={(l) => (l % 5 === 0 ? String(l) : "")}
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
              </Panel>

              <Panel title={<><b>Figure 13.3</b> · the experts that matter</>}
                note="The twelve largest single-expert knockout effects, aggregated over prompts. Most experts are removable at this corpus size; the few that are not are candidates for stored knowledge, not necessarily ‘maths experts’.">
                <div className="stack" style={{ gap: 6 }}>
                  {topExperts.map((e, i) => {
                    const hi = Math.max(...topExperts.map((x) => Math.abs(x.delta)), 1e-6);
                    return (
                      <div key={i} style={{ display: "grid", gridTemplateColumns: "86px 1fr 64px", gap: 8, alignItems: "center" }}>
                        <span className="mono" style={{ fontSize: 10 }}>L{e.l} · E{e.e}</span>
                        <div style={{ height: 10, background: "rgba(16,18,23,0.04)" }}>
                          <div style={{ width: `${(Math.abs(e.delta) / hi) * 100}%`, height: "100%", background: e.delta > 0 ? "var(--ember)" : "var(--teal)" }} />
                        </div>
                        <span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>{e.delta > 0 ? "+" : ""}{fmt(e.delta, 3)}</span>
                      </div>
                    );
                  })}
                </div>
              </Panel>
            </div>

            <Panel title={<><b>Figure 13.4</b> · the shared experts</>}
              note="DeepSeek routes every token through shared experts in addition to the six routed ones. Removing one shared expert is a much larger intervention than removing one of 64 routed experts — the comparison is the point.">
              <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
                {ab.shared.map((s, i) => (
                  <div key={i} className="stat" style={{ minWidth: 150 }}>
                    <div className="k">shared expert · L{s.l}</div>
                    <div className="v" style={{ color: s.delta > 0 ? "var(--ember)" : "var(--teal)" }}>{s.delta > 0 ? "+" : ""}{fmt(s.delta, 2)}</div>
                    <div className="s">Δ NLL, mean over prompts</div>
                  </div>
                ))}
                <div className="stat" style={{ minWidth: 150 }}>
                  <div className="k">routed expert median</div>
                  <div className="v">{fmt([...ab.experts].sort((a, b) => a.delta - b.delta)[Math.floor(ab.experts.length / 2)]?.delta ?? 0, 3)}</div>
                  <div className="s">for comparison</div>
                </div>
              </div>
            </Panel>

            <Prose>
              <p>
                The knockout curve is the causal counterpart to the anatomy page, but it is
                not monotonic. L0 and L26 dominate with opposite signs: zeroing L26's MLP
                raises NLL by about 3.25 nats, while the L0 interventions lower it by about
                4 nats. Middle layers are mostly small. The large L0 effect is a warning
                that zeroing a sublayer is not a clean “remove information” operation at the
                first block, not evidence that early computation is unnecessary.
              </p>
              <p>
                The expert bars are mostly small. That is expected: with 64 experts and top-6
                routing, removing one expert affects a fraction of tokens and the gate can
                compensate. The shared experts are different — every token uses them, so
                their knockout is a global intervention. This is the strongest argument that
                the shared expert is doing generic work that routing does not replace.
              </p>
            </Prose>

            <Method>
              <p>
                Ablation sets a sublayer's output to zero, preserving the residual path and
                the rest of the block; whole-block ablation replaces the block with its input,
                removing it from the residual path. NLL is the negative log-probability of
                the target continuation, mean over prompts, in nats.
              </p>
              <p>
                Expert ablation zeroes the selected expert's output before the weighted sum;
                the gate weights are unchanged, so this tests the expert's contribution, not
                a rerouting. Shared experts are ablated one at a time in the same way.
              </p>
            </Method>

            <Footnotes items={[
              `Baseline NLL ${fmt(ab.baseline, 3)} on ${ab.prompts.length} prompts, ${meta?.modelName ?? "V2-Lite"} in ${meta?.precision ?? "NF4"}.`,
              "A null ablation is not proof of irrelevance: the model may compensate through another path, especially with a shared expert.",
              "Per-prompt effects are noisy at this corpus size; the aggregate curve is more reliable than any single cell.",
              "The sign convention is Δ = ablated − baseline, so positive is worse. A negative value means the component reduced the target probability.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={13} />
    </>
  );
}
