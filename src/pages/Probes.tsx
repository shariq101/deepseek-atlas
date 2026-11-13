import React, { useMemo, useState } from "react";
import { Meta, Probes, fmt, pct, useMeasure, useOne, SEQ, ramp } from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart } from "../components/charts";

const PALETTE = ["var(--signal)", "var(--ember)", "var(--teal)", "var(--olive)", "var(--plum)", "var(--sand)", "var(--ink-2)"];

export default function ProbesPage() {
  const metaQ = useOne<Meta>("meta");
  const prQ = useOne<Probes>("probes");
  const [prop, setProp] = useState<string | null>(null);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const pr = prQ.data;
  const L = pr?.layers.length ?? meta?.layers ?? 27;
  const properties = pr?.properties ?? [];
  const active = properties.find((p) => p.key === prop) ?? properties[0];

  const final = useMemo(() => {
    if (!pr) return [];
    return properties.map((p) => ({
      p, acc: pr.byProp[p.key]?.[pr.byProp[p.key].length - 1] ?? 0,
    })).sort((a, b) => b.acc - a.acc);
  }, [pr, properties]);

  return (
    <>
      <ChapterHead n="16" part="II" kicker="Probes"
        title={<>What is <em>readable</em> at each depth</>}
        standfirst={<>A linear probe asks a narrow question: is this property linearly decodable from the residual? It is not a claim that the model uses the property. But a property that appears at L4 and disappears at L20 is telling you when the representation is still local and when it has been abstracted away.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>{properties.length} properties</Chip><Chip>logistic probe</Chip></>}
      />
      <DataState loading={metaQ.loading || prQ.loading} error={metaQ.error ?? prQ.error} what="the probe sweep">
        {pr && (
          <>
            <div className="grid4">
              <Stat k="Properties" v={properties.length} s="binary / multiclass" />
              <Stat k="Positions" v={pr.positions} s="per prompt" />
              <Stat k="Prompts" v={pr.nPrompts} s="probe corpus" />
              <Stat k="Best final" v={final[0] ? pct(final[0].acc, 0) : "—"} s={final[0]?.p.label ?? ""} />
            </div>

            <Panel title={<><b>Figure 16.1</b> · decodability through depth</>}
              tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>chance-corrected</span>}
              note="Each line is a property, plotted as accuracy minus chance; 0 is therefore chance, not a dashed reference series. The layer where a curve peaks is where that property is most linearly available.">
              <div ref={ref}>
                <LineChart width={Math.max(width, 380)} height={300}
                  series={properties.map((p, i) => ({
                    values: (pr.byProp[p.key] ?? []).map((v) => v - p.chance),
                    color: PALETTE[i % PALETTE.length], label: p.label, width: 1.7,
                  }))}
                  xTicks={[0, 7, 14, 20, L - 1]} xLabel="layer" yLabel="accuracy − chance"
                  yFormat={(v) => (v > 0 ? "+" : "") + pct(v, 0)} />
              </div>
              <div className="legend" style={{ marginTop: 6 }}>
                <Legend items={properties.map((p, i) => ({ c: PALETTE[i % PALETTE.length], label: p.label, line: true }))} />
              </div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 16.2</b> · where each property lives</>}
                note="Rows are properties, columns layers; colour is accuracy above chance. Horizontal bands are properties that are readable everywhere; isolated bright cells are transient representations.">
                <Heat rows={properties.length} cols={L} cellW={Math.max(10, 420 / L)} cellH={16} gap={1}
                  value={(p, l) => {
                    const prop = properties[p];
                    const acc = pr.byProp[prop.key]?.[l] ?? 0;
                    return Math.max(0, Math.min(1, (acc - prop.chance) / Math.max(1 - prop.chance, 1e-6)));
                  }}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(p) => properties[p].label.slice(0, 14)}
                  colLabels={(l) => (l % 4 === 0 ? String(l) : "")}
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
              </Panel>

              <Panel title={<><b>Figure 16.3</b> · the final readout</>}
                note="Final-layer accuracy for each property against its chance rate. A property can be highly decodable at the end and still be unused by the model; probing measures availability, not necessity.">
                <div className="stack" style={{ gap: 7 }}>
                  {final.map(({ p, acc }, i) => (
                    <div key={p.key} style={{ display: "grid", gridTemplateColumns: "118px 1fr 44px 52px", gap: 7, alignItems: "center" }}>
                      <span style={{ fontSize: 10.5, color: "var(--ink-2)" }}>{p.label}</span>
                      <div style={{ height: 11, background: "rgba(16,18,23,0.04)", position: "relative" }}>
                        <div style={{ width: `${acc * 100}%`, height: "100%", background: PALETTE[i % PALETTE.length] }} />
                        <div style={{ position: "absolute", left: `${p.chance * 100}%`, top: -2, width: 1, height: 15, background: "var(--ink)" }} title={`chance ${pct(p.chance, 0)}`} />
                      </div>
                      <span className="mono" style={{ fontSize: 9.5 }}>{pct(acc, 0)}</span>
                      <span className="mono" style={{ fontSize: 8.5, color: "var(--ink-3)" }}>ch {pct(p.chance, 0)}</span>
                    </div>
                  ))}
                </div>
              </Panel>
            </div>

            <div className="grid4">
              <Stat k="Peak layer" v={active ? `L${(pr.byProp[active.key] ?? []).indexOf(Math.max(...(pr.byProp[active.key] ?? [0])))}` : "—"} s={active?.label ?? ""} />
              <Stat k="Peak accuracy" v={active ? pct(Math.max(...(pr.byProp[active.key] ?? [0])), 0) : "—"} s={`chance ${active ? pct(active.chance, 0) : "—"}`} />
              <Stat k="Features" v="2048" s="residual width" />
              <Stat k="Train/test" v="5-fold" s="grouped by prompt" />
            </div>

            <Prose>
              <p>
                Not every property behaves the same way. Surface properties — is the token
                punctuation, is it a digit, is it inside a code block — are readable almost
                from the embedding and fade only slightly. Structural properties — is this
                the answer token, is this position part of a repeated pattern — rise in the
                middle layers and peak where the model has assembled enough context.
              </p>
              <p>
                The important negative result is that high probe accuracy does not imply
                use. A probe can read a property the model never consults for this task.
                Chapter 13 is the corrective: it deletes the component that carries the
                property and measures whether the output changes.
              </p>
            </Prose>

            <Method>
              <p>
                For each layer and property, a logistic regression is trained on the
                residual state at every position of the probe corpus, with 5-fold
                cross-validation grouped by prompt so that positions from the same prompt
                never appear in both train and test. Accuracy is reported on held-out folds
                and corrected by the property's chance rate.
              </p>
              <p>
                Features are the raw 2048-d residual, standardised per layer, then reduced
                to 32 components by PCA fitted inside each training fold. The probe is
                deliberately simple so that a positive result means linear readability,
                not probe capacity.
              </p>
            </Method>

            <Footnotes items={[
              `Measured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); ${pr.nPrompts} prompts, ${pr.positions} positions.`,
              "Grouped cross-validation is essential: random splits leak prompt identity and inflate accuracy.",
              "Chance is the majority-class rate for binary properties and 1/k for balanced multiclass; the bar is not always 50%.",
              "A linear probe cannot detect a property encoded nonlinearly; a null result is not evidence of absence.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={16} />
    </>
  );
}
