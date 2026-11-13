import React, { useMemo } from "react";
import { ContextData, Meta, fmt, pct, useMeasure, useOne, SEQ, ramp } from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Legend, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat, LineChart, Scatter } from "../components/charts";

const COLORS = ["var(--signal)", "var(--ember)", "var(--teal)", "var(--olive)", "var(--plum)", "var(--sand)"];

export default function ContextPage() {
  const metaQ = useOne<Meta>("meta");
  const ctxQ = useOne<ContextData>("context");
  const { ref, width } = useMeasure<HTMLDivElement>();
  const meta = metaQ.data;
  const ctx = ctxQ.data;

  const lengths = useMemo(() => Array.from(new Set(ctx?.results.map((r) => r.len) ?? [])).sort((a, b) => a - b), [ctx]);
  const depths = useMemo(() => Array.from(new Set(ctx?.results.map((r) => r.depth) ?? [])).sort((a, b) => a - b), [ctx]);

  const series = useMemo(() => depths.map((d, i) => ({
    values: lengths.map((l) => {
      const rows = ctx?.results.filter((r) => r.len === l && r.depth === d) ?? [];
      return rows.reduce((a, b) => a + b.targetProb, 0) / Math.max(rows.length, 1);
    }),
    color: COLORS[i % COLORS.length], label: `depth ${pct(d, 0)}`, width: 1.7,
  })), [ctx, lengths, depths]);

  const overall = ctx ? ctx.results.reduce((a, b) => a + b.targetProb, 0) / Math.max(ctx.results.length, 1) : 0;
  // Row-normalised attention is dominated by a few positions; a square-root
  // transfer keeps the small non-zero masses visible without changing the data.
  const attentionShade = (v: number) => v <= 0 ? 0 : 0.14 + 0.86 * Math.sqrt(v);

  return (
    <>
      <ChapterHead n="17" part="II" kicker="Long context"
        title={<>What survives four <em>thousand</em> tokens</>}
        standfirst={<>A long-context model is not one capability but two: the ability to attend to the right token, and the ability to keep the cache affordable while doing it. This chapter measures retrieval on V2-Lite to 4,096 tokens; the million-token cache curves are config arithmetic, labelled as such.</>}
        chips={<><Chip tone="teal">measured · 4k</Chip><Chip>analytic · 1M</Chip><Chip>V2-Lite · 27 layers</Chip></>}
      />
      <DataState loading={metaQ.loading || ctxQ.loading} error={metaQ.error ?? ctxQ.error} what="the retrieval capture">
        {ctx && (
          <>
            <div className="grid4">
              <Stat k="Mean target P" v={pct(overall, 1)} s="15 measured runs" />
              <Stat k="Lengths" v={lengths.length} s={`${lengths[0] ?? 0}–${lengths[lengths.length - 1] ?? 0} tokens`} />
              <Stat k="Depths" v={depths.length} s="needle positions" />
              <Stat k="Runs" v={ctx.results.length} s="forward passes" />
            </div>

            <Panel title={<><b>Figure 17.1</b> · retrieval probability by length</>}
              note="Each line is a needle depth (fraction of the context before the fact); the y value is the model's probability of the target answer token. A flat line is robust retrieval; a falling line is context dilution. The sampled lengths double, so their index spacing is logarithmic.">
              <div ref={ref}>
                <LineChart width={Math.max(width, 380)} height={300}
                  series={series}
                  xTicks={lengths.map((_, i) => i)}
                  xFormat={(i) => `${Math.round((lengths[i] ?? 0) / 1024)}k`}
                  yDomain={[0, 1]} xLabel="context length (log)" yLabel="P(target)"
                  yFormat={(v) => pct(v, 0)} />
              </div>
              <div className="legend" style={{ marginTop: 6 }}>
                <Legend items={depths.map((d, i) => ({ c: COLORS[i % COLORS.length], label: `depth ${pct(d, 0)}`, line: true }))} />
              </div>
            </Panel>

            <div className="grid2">
              <Panel title={<><b>Figure 17.2</b> · the retrieval surface</>}
                note="Every measured (length, depth) cell, coloured by target probability. A dip in the middle would be the ‘lost in the middle’ effect; a uniform field means position barely matters. This capture is close to uniform, with a small decline at the longest length.">
                <Heat rows={depths.length} cols={lengths.length} cellW={Math.max(22, 320 / Math.max(lengths.length, 1))} cellH={22} gap={2}
                  value={(d, l) => {
                    const rows = ctx.results.filter((r) => r.depth === depths[d] && r.len === lengths[l]);
                    return rows.length ? rows.reduce((a, b) => a + b.targetProb, 0) / rows.length : 0;
                  }}
                  color={(v) => ramp(v, SEQ)}
                  rowLabels={(d) => pct(depths[d], 0)}
                  colLabels={(l) => `${Math.round((lengths[l] ?? 0) / 1024)}k`}
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`} />
              </Panel>

              <Panel title={<><b>Figure 17.3</b> · where the final query attends (512 tokens)</>}
                note="Mean attention across heads from the final query to every key position, at the one captured 512-token length. Rows are layers, columns key positions; each row is normalised to its own maximum. Colour uses a square-root transfer so small non-zero masses remain visible. This is one length, not a length sweep.">
                {ctx.attention ? (
                  <div style={{ overflowX: "auto" }}>
                    <Heat rows={ctx.attention.meanByLayer.length} cols={ctx.attention.meanByLayer[0]?.length ?? 0} cellW={2.4} cellH={10} gap={0.2}
                      value={(l, c) => {
                        const row = ctx.attention!.meanByLayer[l] ?? [];
                        const hi = Math.max(...row, 1e-6);
                        return (row[c] ?? 0) / hi;
                      }}
                      color={(v) => ramp(attentionShade(v), SEQ)}
                      rowLabels={(l) => (l % 5 === 0 ? String(l) : "")}
                      colLabels={(c) => (c % 64 === 0 ? String(c) : "")}
                      formatValue={(v) => `${(v * 100).toFixed(0)}% of row max`} />
                  </div>
                ) : <div className="caption">Attention maps were not captured for this run.</div>}
                <div className="caption" style={{ marginTop: 6 }}>
                  Colour uses a square-root scale: paper is zero, ink is the row maximum.
                  The transfer lifts faint attention above the page tint; read the value in the tooltip,
                  not the absolute darkness alone.
                </div>
              </Panel>
            </div>

            <div className="grid4">
              <Stat k="P(target) at capture" v={ctx.attention ? pct(ctx.attention.peak, 1) : "—"} s="512-token run" />
              <Stat k="Longest" v={`${Math.round((lengths[lengths.length - 1] ?? 0) / 1024)}k`} s="tokens measured" />
              <Stat k="Worst depth" v={(() => {
                const byDepth = depths.map((d) => ({ d, p: ctx.results.filter((r) => r.depth === d).reduce((a, b) => a + b.targetProb, 0) }));
                const w = byDepth.sort((a, b) => a.p - b.p)[0];
                return w ? pct(w.d, 0) : "—";
              })()} s="lowest recall" />
              <Stat k="Method" v="needle" s="single fact, one query" />
            </div>

            <Prose>
              <p>
                Across the measured 256–4,096-token range, retrieval stays high: the
                target probability is 0.94–0.98, with a small decline at the longest
                length and no strong middle-position penalty in this one template. That
                is a narrower result than “lost in the middle”, and the 1M-token claims
                on the frontier page are config arithmetic, not a continuation of it.
              </p>
              <p>
                The cache is the other half. V2-Lite's MLA stores 30.4 KiB per token at
                fp16, 7.1× less than a 16-head MHA model would. That is what makes a
                longer window feasible on a T4; the capture budget, not the model's
                configured maximum, is why the measured sweep stops at 4k.
              </p>
            </Prose>

            <Method>
              <p>
                A single fact is inserted at a controlled depth in a repeated distractor
                context; the prompt asks for the fact's value. The target probability is the
                model's probability of the correct answer's first token. Lengths are powers
                of two from 256 to 4,096 tokens; depths are fractions of the context.
              </p>
              <p>
                The attention panel is the final query's distribution over key positions,
                averaged over query heads, for the single 512-token capture. It is not a
                needle-mass-by-length summary; the capture stores one attention map, not a
                map per length.
              </p>
            </Method>

            <Footnotes items={[
              `Measured on ${meta?.modelName ?? "V2-Lite"} (${meta?.precision ?? "NF4"}); one fact per prompt, no multi-fact reasoning. ${ctx.method}`,
              "A single-needle task measures retrieval, not long-context reasoning. Multi-fact and aggregation tasks are harder and are not captured here.",
              "The measured sweep stops at 4,096 tokens because that was the capture budget, not the checkpoint's maximum. One million tokens is config arithmetic.",
              "Cache figures for V3/V4 are analytic from configs; only V2-Lite's are measured.",
            ]} />
          </>
        )}
      </DataState>
      <PageFoot index={17} />
    </>
  );
}
