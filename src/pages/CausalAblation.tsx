import React, { useMemo, useState } from "react";
import {
  CausalHeadRow, CausalInterventions, clamp, fmt, ramp, useOne,
} from "../lib/atlas";
import {
  ChapterHead, Chip, DataState, Footnotes, Method, PageFoot, Panel, Prose, Stat,
} from "../components/ui";
import { Heat } from "../components/charts";

const HEAD_LAYERS = 27;
const HEAD_COUNT = 16;

export default function CausalAblation() {
  const q = useOne<CausalInterventions>("interventions");
  const data = q.data;
  const [head, setHead] = useState<{ r: number; c: number } | null>(null);

  const headGrid = useMemo(() => {
    const grid = Array.from({ length: HEAD_LAYERS }, () =>
      Array.from({ length: HEAD_COUNT }, () => 0),
    );
    for (const row of data?.headRows ?? []) {
      if (grid[row.l] && grid[row.l][row.h] !== undefined) grid[row.l][row.h] = row.deltaProb;
    }
    return grid;
  }, [data]);

  const expertGrid = useMemo(() => {
    const grid = Array.from({ length: HEAD_LAYERS }, () =>
      Array.from({ length: 8 }, () => 0),
    );
    for (const row of data?.expertRows ?? []) {
      const col = [0, 7, 15, 23, 31, 39, 47, 55].indexOf(row.e);
      if (col >= 0 && grid[row.l]) grid[row.l][col] = row.deltaProb;
    }
    return grid;
  }, [data]);

  const maxHead = Math.max(1e-9, ...headGrid.flat().map((v) => Math.abs(v)));
  const maxExpert = Math.max(1e-9, ...expertGrid.flat().map((v) => Math.abs(v)));
  const selected = data?.headRows.find((r) => r.l === head?.r && r.h === head?.c);
  const topHead = data?.topHeads?.[0];
  const topExpert = data?.topExperts?.[0];

  return (
    <>
      <ChapterHead
        n="22" part="IV" kicker="Causal interventions"
        title={<>The model, <em>minus a head</em></>}
        standfirst={
          <>
            The pages before this one compare a readout graph with an offline proxy. This
            page is the other kind of evidence: we remove one attention head or one routed
            expert from V2-Lite, rerun the model, and record the target probability after
            the intervention. The damage is measured, not inferred from weights.
          </>
        }
        chips={
          <>
            <Chip tone="blue">real forward passes</Chip>
            <Chip>V2-Lite · NF4</Chip>
            <Chip tone="teal">4 prompts</Chip>
          </>
        }
      />

      <div className="grid3" style={{ marginBottom: 24 }}>
        <Stat k="Interventions" v={data ? `${data.headRows.length} heads · ${data.expertRows.length} experts` : "loading"} s="each one a separate forward pass" />
        <Stat k="Largest head effect" v={topHead ? fmt(Math.abs(topHead.deltaProb), 3) : "—"} s={topHead ? `L${topHead.l} · H${topHead.h} · Δ target prob` : "awaiting model"} />
        <Stat k="Largest expert effect" v={topExpert ? fmt(Math.abs(topExpert.deltaProb), 3) : "—"} s={topExpert ? `L${topExpert.l} · expert ${topExpert.e}` : "awaiting model"} />
      </div>

      <Prose>
        <p>
          A causal intervention asks a different question from a correlation. The
          <strong> readout graph</strong> says where a component writes toward the answer;
          a <strong>head ablation</strong> says what happens to the answer when that write is
          removed. The two can disagree. That disagreement is the useful part: it is the
          difference between a promising address and a load-bearing part.
        </p>
        <p className="aside">
          This is still one intervention family, not a complete circuit proof. Zeroing a
          head at every token position changes the context available to later layers, and
          the model may compensate through other heads. A causal claim of the form
          “this head carries the behaviour” needs a matching intervention on the path, not
          only a drop in a single metric. The map below is a hit list for those experiments.
        </p>
      </Prose>

      <DataState loading={q.loading} error={q.error} what="the causal intervention capture">
        {data && (
          <>
            <Panel
              title={<><b>Figure 22.1</b> · zero-ablate one head, rerun the model</>}
              note={`Mean change in target probability across ${data.prompts.length} prompts after zeroing a single head's slice at o_proj. Rows are layers, columns are the ${data.heads} query heads. Darker blue means a larger absolute change; the sign is in the detail below. Every intervention is a real forward pass on ${data.model} (${data.quant}).`}
            >
              <div style={{ overflowX: "auto", paddingBottom: 4 }}>
                <Heat
                  rows={data.layers} cols={data.heads}
                  cellW={17} cellH={13} gap={1}
                  value={(l, h) => headGrid[l]?.[h] ?? 0}
                  color={(v) => ramp(clamp(Math.abs(v) / maxHead, 0, 1), [
                    [0, "#f3f1ea"], [0.3, "#dfe4f8"], [0.7, "#8d9ee6"], [1, "#15246e"],
                  ])}
                  rowLabels={(l) => `L${String(l).padStart(2, "0")}`}
                  colLabels={(h) => String(h)}
                  onHover={(l, h) => setHead({ r: l, c: h })}
                  highlight={head}
                  formatValue={(v) => `Δp ${fmt(v, 4)}`}
                />
              </div>
              <div className="hoverbox" style={{ marginTop: 8 }}>
                {selected ? (
                  <>
                    layer <b>{selected.l}</b> · head <b>{selected.h}</b> ·
                    Δ target probability <b>{fmt(selected.deltaProb, 4)}</b> ·
                    Δ margin <b>{fmt(selected.deltaMargin, 3)}</b> ·
                    per prompt {selected.perPrompt.map((v) => fmt(v, 3)).join(" · ")}
                  </>
                ) : (
                  "Hover a cell to read the head, its target-probability change and its per-prompt spread."
                )}
              </div>
            </Panel>

            <div className="grid2" style={{ alignItems: "start" }}>
              <Panel title={<><b>Figure 22.2</b> · the largest head effects</>} tight
                note="Ranked by absolute mean change in target probability. A negative value is damage; a positive value means the removal made the target slightly more likely, which is evidence of compensation rather than importance.">
                <table className="tbl">
                  <thead><tr><th>layer</th><th>head</th><th>Δ target p</th><th>Δ margin</th></tr></thead>
                  <tbody>
                    {(data.topHeads ?? []).slice(0, 10).map((r: CausalHeadRow) => (
                      <tr key={`${r.l}-${r.h}`}>
                        <td className="k">L{r.l}</td><td>H{r.h}</td>
                        <td>{fmt(r.deltaProb, 4)}</td><td>{fmt(r.deltaMargin, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Panel>
              <Panel title={<><b>Figure 22.3</b> · routed experts, removed one at a time</>} tight
                note={`Eight fixed routed experts across every MoE layer, on ${data.prompts.length} prompts. Rows are layers; columns are the fixed expert ids below. The expert result is a separate causal check from the existing layer sweep.`}>
                <div style={{ overflowX: "auto" }}>
                  <Heat
                    rows={data.layers} cols={8} cellW={26} cellH={13} gap={1}
                    value={(l, c) => expertGrid[l]?.[c] ?? 0}
                    color={(v) => ramp(clamp(Math.abs(v) / maxExpert, 0, 1), [
                      [0, "#f3f1ea"], [0.3, "#f0dbcd"], [0.7, "#d07a54"], [1, "#7d2a10"],
                    ])}
                    rowLabels={(l) => `L${String(l).padStart(2, "0")}`}
                    colLabels={(c) => String([0, 7, 15, 23, 31, 39, 47, 55][c])}
                    formatValue={(v) => `Δp ${fmt(v, 4)}`}
                  />
                </div>
              </Panel>
            </div>

            <Prose>
              <p className="aside">
                The strongest heads are not necessarily the heads with the largest
                attention mass. That is the same distinction the Mechanism page makes:
                attention to a token is not the same as a causal write to the answer.
                The top of this table is a shortlist for path patching, not a finished
                mechanism. The full per-layer table is in <code>interventions.json</code>.
              </p>
            </Prose>
          </>
        )}
      </DataState>

      <Method label="How the intervention was run">
        <p>
          The model is DeepSeek-V2-Lite at NF4 with fp16 compute on one T4. For each head,
          a forward pre-hook zeros only that head's slice of the input to <code>o_proj</code>;
          for each routed expert, a hook zeros that expert's output. The metric is the
          target token's probability at the final position, with the margin over the best
          non-target token recorded alongside it. The four prompts are the deep-dive prompts
          used elsewhere in the atlas: fact-capital, syn-attract, ind-1 and code-add.
        </p>
        <p>
          This is zero ablation, not mean ablation. It is a real model forward pass after
          every intervention, but it is not path patching and it does not establish
          sufficiency. Run with <code>STAGES=interventions</code> in the capture kernel.
        </p>
      </Method>

      <Footnotes items={[
        "The capture is a single 4-bit V2-Lite checkpoint. A head that is redundant under NF4 may be less redundant at bf16; the direction of the effect is the useful result, not the absolute probability.",
        "Each row is the mean over four prompts; perPrompt in interventions.json preserves the spread. The target probability baseline is shown in the panel tooltip.",
        "Zeroing a head at every position is a strong intervention. It removes the head's contribution to all later token states, not only the final token.",
        "A positive delta does not mean the component is harmful in general; it can reflect compensation or a metric that is not the behaviour of interest.",
      ]} />
      <PageFoot index={22} />
    </>
  );
}
