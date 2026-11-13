import React, { useMemo, useState } from "react";
import {
  Experts, Meta, fmt, pct, useMeasure, useOne, SEQ, ramp,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart, Scatter } from "../components/charts";

const GROUP_COLORS = ["var(--signal)", "var(--ember)", "var(--teal)", "var(--olive)", "var(--plum)", "var(--sand)", "var(--ink-3)", "var(--ink-2)"];

export default function ExpertsPage() {
  const metaQ = useOne<Meta>("meta");
  const exQ = useOne<Experts>("experts");
  const [layer, setLayer] = useState(12);
  const [hover, setHover] = useState<number | null>(null);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const ex = exQ.data;

  const atLayer = useMemo(() => ex?.profiles.filter((p) => p.l === layer) ?? [], [ex, layer]);
  const groups = ex?.groups ?? [];
  const gColor = (g: string) => GROUP_COLORS[Math.max(0, groups.indexOf(g)) % GROUP_COLORS.length];

  const summary = useMemo(() => {
    if (!ex) return null;
    const flat = ex.profiles.filter((p) => p.l > 0);
    const sorted = [...flat].sort((a, b) => (b.topGroups[0]?.[1] ?? 0) - (a.topGroups[0]?.[1] ?? 0));
    const top = sorted[0];
    // Layer 0 is the dense prefix and has no router; do not count it as a dead
    // expert layer. The heatmap and statistics below cover layers 1..L-1.
    const dead = ex.load.slice(1).flat().filter((v) => v < 1e-6).length;
    const total = ex.load.slice(1).flat().length;
    return { top, dead, total };
  }, [ex]);

  return (
    <>
      <ChapterHead n="10" part="II" kicker="Expert geography"
        title={<>A division of labour, <em>if it exists</em></>}
        standfirst={<>Sparse experts are often described as a committee of specialists. The checkpoint does not label them. What it does contain is a co-activation matrix, a load profile per layer, and the contexts that selected each expert — enough to ask whether specialisation is real, and at what scale.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>{meta?.experts ?? 64} experts</Chip><Chip>{(meta?.layers ?? 27) - 1} MoE layers</Chip></>}
      />
      <DataState loading={metaQ.loading || exQ.loading} error={metaQ.error ?? exQ.error} what="the expert capture">
        {ex && summary && (
          <>
            <Panel title={<><b>Figure 10.1</b> · load across the expert population</>}
              note="Rows are layers 1–26, the 26 MoE layers; the first layer is dense and has no router. Each row is scaled by 2.2× its own mean and clipped, so a dark cell is an expert used above its peer average at that depth.">
              <div style={{ overflowX: "auto" }}>
                <Heat rows={ex.layers - 1} cols={ex.experts} cellW={13} cellH={14} gap={1}
                  value={(l, e) => {
                    const row = ex.load[l + 1] ?? [];
                    const mean = row.reduce((a, b) => a + b, 0) / Math.max(row.length, 1) || 1;
                    return Math.min(1, (row[e] ?? 0) / (mean * 2.2));
                  }}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(l) => String(l + 1)}
                  colLabels={(e) => (e % 8 === 0 ? `E${e}` : "")}
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
              </div>
              <div className="caption">Normalised load. Persistent vertical stripes are experts that are broadly useful; isolated dark cells are layer-specific.</div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 10.2</b> · co-activation</>}
                note="How often two experts are selected by the same token, aggregated over the corpus. Block structure means experts are chosen in teams; a diagonal-only matrix means choices are independent.">
                <Heat rows={ex.experts} cols={ex.experts} cellW={width > 700 ? 5.6 : 4} cellH={width > 700 ? 5.6 : 4} gap={0.5}
                  value={(a, b) => {
                    const v = ex.coactivation[a]?.[b] ?? 0;
                    const hi = Math.max(...ex.coactivation.flat(), 1);
                    return Math.sqrt(v / hi);
                  }}
                  color={(v) => ramp(v, SEQ)}
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
                <div className="caption">64 × 64, symmetric. Diagonal removed by construction.</div>
              </Panel>

              <Panel title={<><b>Figure 10.3</b> · who specialises?</>}
                tools={<span className="mono" style={{ fontSize: 9 }}>L{layer}</span>}
                note="Each dot is one expert at the chosen layer: how many tokens selected it (x) and its mean gate weight (y). Colour is the prompt group that dominates its context profile. The axes are load and weight, not a quality score; colour is a descriptive affiliation.">
                <input type="range" min={1} max={ex.layers - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} style={{ marginBottom: 4 }} />
                <div ref={ref}>
                  <Scatter width={Math.max(width / 2 - 20, 320)} height={260}
                    points={atLayer.map((p, i) => ({
                      x: p.count, y: p.meanWeight, r: hover === i ? 7 : 4.5,
                      color: gColor(p.topGroups[0]?.[0] ?? ""),
                      label: `L${p.l} E${p.e} · ${p.count} tokens · weight ${fmt(p.meanWeight, 3)} · ${p.topGroups.map(([g, s]) => `${g} ${pct(s, 0)}`).join(", ")}`,
                    }))}
                    xLabel="tokens routed" yLabel="mean gate weight"
                    yFormat={(v) => fmt(v, 2)} highlight={hover} onHover={setHover} />
                </div>
                <div className="legend" style={{ marginTop: 6 }}>
                  <Legend items={groups.map((g) => ({ c: gColor(g), label: g }))} />
                </div>
              </Panel>
            </div>

            <div className="grid4">
              <Stat k="Most specialised" v={summary.top ? `L${summary.top.l} · E${summary.top.e}` : "—"} s={summary.top ? `${pct(summary.top.topGroups[0]?.[1] ?? 0, 0)} ${summary.top.topGroups[0]?.[0]}` : ""} />
              <Stat k="Dead cells" v={`${summary.dead} / ${summary.total}`} s="layer × expert" />
              <Stat k="Corpus groups" v={groups.length} s="prompt families" />
              <Stat k="Experts per layer" v={ex.experts} s={`${ex.layers - 1} MoE layers`} />
            </div>

            <h2 className="sec">Specialisation is real but shallow</h2>
            <Prose>
              <p>
                The co-activation matrix is the cleanest evidence. If experts were
                independent, it would be nearly diagonal after removing self-pairs. It is
                not: there are blocks of experts that fire together in the aggregate.
                The matrix has no layer axis, so this page cannot claim those blocks recur
                across layers; that would require a per-layer co-activation capture. The
                aggregate is consistent with a learned partition of the residual space,
                not with 64 copies of the same function.
              </p>
              <p>
                The profiles are weaker evidence. An expert's “top group” is the prompt
                family that selected it most often, but the margin is usually small, and
                the same expert appears in several families. Calling E17 “the maths
                expert” would overfit 27 prompts. The honest claim is narrower: routing is
                input-dependent, reproducible across prompts of the same type, and
                measurable — not that each expert has a human-readable topic.
              </p>
            </Prose>

            <Method>
              <p>
                Load counts top-{meta?.topk ?? 6} selections over all tokens and prompts.
                Co-activation counts pairs selected by the same token in the same layer,
                normalised by the maximum pair count. Profiles aggregate by prompt group;
                mean weight is the average gate weight when the expert is selected.
              </p>
              <p>
                The group labels come from the corpus design (factual, arithmetic, code,
                induction, etc.), not from the model. They are a probe, not a taxonomy.
              </p>
            </Method>

            <Footnotes items={[
              `Captured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); ${meta?.corpus ?? 27} prompts. Expert indices are checkpoint order.`,
              "Co-activation is corpus-dependent; a different prompt distribution would change it. The matrix is descriptive, not a property of the weights alone.",
              "Dead cells may be rare at this corpus size rather than unused in general. A 27-prompt corpus cannot estimate tail usage.",
              "Group affinity is computed only where an expert was selected; it does not measure what the expert would do if forced.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={10} />
    </>
  );
}
