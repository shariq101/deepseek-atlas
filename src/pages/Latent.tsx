import React, { useMemo, useState } from "react";
import { Latent, Meta, fmt, pct, useMeasure, useOne, SEQ, ramp } from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart, Scatter } from "../components/charts";

export default function LatentPage() {
  const metaQ = useOne<Meta>("meta");
  const latQ = useOne<Latent>("latent");
  const [layer, setLayer] = useState(14);
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const lat = latQ.data;
  const L = lat?.layers ?? meta?.layers ?? 27;

  const normMean = useMemo(
    () => lat?.norms.map((row) => row.reduce((a, b) => a + b, 0) / Math.max(row.length, 1)) ?? [],
    [lat],
  );
  const ropeMean = useMemo(
    () => lat?.ropeNorm.map((row) => row.reduce((a, b) => a + b, 0) / Math.max(row.length, 1)) ?? [],
    [lat],
  );

  return (
    <>
      <ChapterHead n="12" part="II" kicker="Latent geometry"
        title={<>The 512 numbers the cache <em>actually keeps</em></>}
        standfirst={<>MLA does not store 16 heads of keys and values. It stores one 512-dimensional latent per token, plus a 64-dimensional rope key. Everything attention does is a learned linear read of that latent — so its geometry is the model's real working memory.</>}
        chips={<><Chip tone="teal">measured</Chip><Chip>512 + 64 dims</Chip><Chip>27 layers</Chip></>}
      />
      <DataState loading={metaQ.loading || latQ.loading} error={metaQ.error ?? latQ.error} what="the latent capture">
        {lat && (
          <>
            <div className="grid2">
              <Panel title={<><b>Figure 12.1</b> · how many dimensions are really used</>}
                note="Effective rank of the 27 prompt-level latent states per layer, and the share of variance captured by the leading component. With n=27, rank cannot exceed 27; read it as a corpus statistic, not as a capacity claim about the 512-d cache.">
                <div ref={ref}>
                  <LineChart width={Math.max(width / 2 - 20, 320)} height={250}
                    series={[
                      { values: lat.effectiveRank, color: "var(--ink)", label: "effective rank", width: 2 },
                      { values: lat.varShare.map((v) => v * 512), color: "var(--ember)", label: "top-var × 512", width: 1.4, dash: "3 3" },
                    ]}
                    xTicks={[0, 7, 14, 20, L - 1]} xLabel="layer" yLabel="dimensions"
                    yFormat={(v) => fmt(v, 0)} />
                </div>
                <div className="legend" style={{ marginTop: 6 }}>
                  <Legend items={[{ c: "var(--ink)", label: "effective rank", line: true }, { c: "var(--ember)", label: "variance share (rescaled)", line: true }]} />
                </div>
              </Panel>

              <Panel title={<><b>Figure 12.2</b> · representation similarity (CKA)</>}
                note="Linear CKA between the latents of every pair of layers, computed across the 27 prompt-level latents. Blocks mean the representation is stable across depth; a sharp off-diagonal change is a reorganisation, not a gradual drift.">
                <Heat rows={L} cols={L} cellW={width > 700 ? 11 : 9} cellH={width > 700 ? 11 : 9} gap={0.5}
                  value={(a, b) => lat.cka[a]?.[b] ?? 0}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(l) => (l % 4 === 0 ? String(l) : "")}
                  colLabels={(l) => (l % 4 === 0 ? String(l) : "")}
                  formatValue={(v) => fmt(v, 2)} />
              </Panel>
            </div>

            <Panel title={<><b>Figure 12.3</b> · which dimensions carry the signal</>}
              note="Per-dimension variance of the latent (row-normalised), layers × 512. The banding is not random: a small set of dimensions dominates, and their identity changes with depth.">
              <div style={{ overflowX: "auto" }}>
                <Heat rows={L} cols={lat.dim} cellW={Math.max(2.4, Math.min(7, 900 / lat.dim))} cellH={11} gap={0.4}
                  value={(l, d) => {
                    const row = lat.dimVar[l] ?? [];
                    const hi = Math.max(...row, 1e-6);
                    return Math.sqrt((row[d] ?? 0) / hi);
                  }}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(l) => String(l)}
                  colLabels={(d) => (d % 64 === 0 ? String(d) : "")}
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
              </div>
              <div className="caption">Row-normalised variance. Vertical structure is a dimension that matters at many depths; horizontal structure is a layer-specific subspace.</div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 12.4</b> · the latent cloud</>}
                tools={<span className="mono" style={{ fontSize: 9 }}>L{layer}</span>}
                note="PCA of the final-position KV latent at the chosen layer, one point per prompt. Colour cycles with prompt identity. Overlap means prompts share a latent geometry at this depth; separation means the representation has already specialised.">
                <input type="range" min={0} max={L - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} />
                <div ref={ref}>
                  <Scatter width={Math.max(width / 2 - 20, 320)} height={260}
                    points={(lat.pca.points[layer] ?? []).map((p, i) => ({
                      x: p[0], y: p[1], r: 3.4,
                      color: `hsl(${(i * 47) % 360} 46% 46%)`,
                      label: `${lat.ids[i] ?? `prompt ${i}`} · PC1 ${fmt(p[0], 2)} · PC2 ${fmt(p[1], 2)}`,
                    }))}
                    xLabel={`PC1 · ${pct(lat.pca.explained[0] ?? 0, 0)}`}
                    yLabel={`PC2 · ${pct(lat.pca.explained[1] ?? 0, 0)}`} />
                </div>
              </Panel>

              <Panel title={<><b>Figure 12.5</b> · norm and rope scale</>}
                note="Mean L2 norm of the latent and of the decoupled rope key per layer. The rope key is small and nearly constant; the latent grows, then stabilises, mirroring the residual stream.">
                <div ref={ref}>
                  <LineChart width={Math.max(width / 2 - 20, 320)} height={230}
                    series={[
                      { values: normMean, color: "var(--signal)", label: "latent norm", width: 1.8 },
                      { values: ropeMean, color: "var(--teal)", label: "rope norm", width: 1.6 },
                    ]}
                    xTicks={[0, 7, 14, 20, L - 1]} xLabel="layer" yFormat={(v) => fmt(v, 0)} />
                </div>
                <div className="legend" style={{ marginTop: 6 }}>
                  <Legend items={[{ c: "var(--signal)", label: "512-d latent", line: true }, { c: "var(--teal)", label: "64-d rope", line: true }]} />
                </div>
              </Panel>
            </div>

            <div className="grid4">
              <Stat k="Dim" v={lat.dim} s="latent width" />
              <Stat k="Effective rank" v={fmt(lat.effectiveRank[Math.floor(L / 2)] ?? 0, 0)} s={`at L${Math.floor(L / 2)}`} />
              <Stat k="Top variance" v={pct(lat.varShare[Math.floor(L / 2)] ?? 0, 0)} s="leading component" />
              <Stat k="CKA L0→Llast" v={fmt(lat.cka[0]?.[L - 1] ?? 0, 2)} s="linear similarity" />
            </div>

            <h2 className="sec">Compression is not free, but it is structured</h2>
            <Prose>
              <p>
                The latent's effective rank is low on this corpus, but with only 27 prompt
                vectors the estimate cannot exceed 27. That makes it a descriptive corpus
                statistic, not evidence that all 512 directions are intrinsically
                redundant. The dim-variance map is the coordinate-level view: a handful of
                dimensions carry most of the scale, and their identity rotates with depth.
              </p>
              <p>
                CKA is the check on the “layers are a smooth trajectory” story. Some
                adjacent layers are nearly identical; others are separated by a sharp drop.
                Those drops are where the model changes what the latent represents. Geometry
                and readout are separate measurements; this page does not establish that the
                drops line up with logit-lens commitment.
              </p>
            </Prose>

            <Method>
              <p>
                The latent is captured by hooking the input to <code>kv_b_proj</code> (after
                the KV normalisation), so it is the tensor attention actually reads. Effective
                rank is computed across the 27 prompt-level latents at each layer. CKA is
                linear CKA over those same prompt-level latents.
              </p>
              <p>
                PCA is fitted independently per layer across the 27 final-position latents;
                the explained-variance percentages are per layer, not global. Rope norms are
                measured on the 64-d decoupled key before attention.
              </p>
            </Method>

            <Footnotes items={[
              `Measured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); ${lat.ids.length} deep prompts. Layer 0 is the embedding, which has no latent.`,
              "Effective rank depends on centring and on the prompt set; it is a lower bound on the dimensionality needed for these contexts, not a capacity claim.",
              "CKA is invariant to linear transforms but not to the nonlinearities between layers; a low value is not proof of a representational break.",
              "PCA axes are arbitrary in sign and rotation within a tied subspace; read distances, not coordinates.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={12} />
    </>
  );
}
