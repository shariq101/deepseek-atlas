#!/usr/bin/env python3
"""
mechanistic.py — offline mechanistic audit over the captured V2-Lite artefacts.

This is deliberately not a new model run. It reads the existing JSON captures and
asks two questions the atlas already has evidence for:

1. Do direct logit attribution and causal ablation rank the same layers?
   (mechanism.json)
2. If we treat the stored wiring graph as a small linear readout system, what
   happens when ACDC's greedy edge-pruning loop is run on it? (acdc.json)

The ACDC pilot is a *readout* pilot, not a model-edge-ablation run. It is useful
for testing the graph arithmetic and the stability of a candidate circuit; it
cannot replace an actual mean-ablation run with forward passes.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "public" / "data"
BUDGET = 96


def read(name: str):
    return json.loads((DATA / f"{name}.json").read_text())


def mean_abs(values):
    return mean(abs(float(v)) for v in values)


def ranks(values):
    """Average ranks, so ties do not favour an arbitrary layer."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        r = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if not vx or not vy:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def top_overlap(x, y, k=5):
    a = set(sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:k])
    b = set(sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:k])
    return len(a & b)


def build_mechanism():
    dla = read("dla")
    ablation = read("ablation")
    layers = len(ablation["layers"])
    prompts = list(dla)

    attn = [mean_abs([dla[p]["attn"][l] for p in prompts]) for l in range(layers)]
    mlp = [mean_abs([dla[p]["mlp"][l] for p in prompts]) for l in range(layers)]
    abl_attn = [abs(float(x["attn"]["delta"])) for x in ablation["layers"]]
    abl_mlp = [abs(float(x["mlp"]["delta"])) for x in ablation["layers"]]
    abl_block = [abs(float(x["block"]["delta"])) for x in ablation["layers"]]

    def top_rows(values):
        return [
            {"layer": l, "value": round(values[l], 4)}
            for l in sorted(range(layers), key=lambda i: values[i], reverse=True)[:5]
        ]

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "offline comparison of captured direct logit attribution and mean ablation; no new forward passes",
        "nPrompts": len(prompts),
        "nAblationPrompts": len(ablation["prompts"]),
        "layers": layers,
        "agreement": {
            "attnSpearman": round(spearman(attn, abl_attn), 4),
            "mlpSpearman": round(spearman(mlp, abl_mlp), 4),
            "blockSpearman": round(spearman([a + b for a, b in zip(attn, mlp)], abl_block), 4),
            "attnTop5Overlap": top_overlap(attn, abl_attn),
            "mlpTop5Overlap": top_overlap(mlp, abl_mlp),
            "blockTop5Overlap": top_overlap([a + b for a, b in zip(attn, mlp)], abl_block),
        },
        "top": {
            "attnAttribution": top_rows(attn),
            "attnAblation": top_rows(abl_attn),
            "mlpAttribution": top_rows(mlp),
            "mlpAblation": top_rows(abl_mlp),
            "blockAblation": top_rows(abl_block),
        },
        "table": [
            {
                "layer": l,
                "attnDla": round(attn[l], 4),
                "mlpDla": round(mlp[l], 4),
                "attnAblation": round(abl_attn[l], 4),
                "mlpAblation": round(abl_mlp[l], 4),
                "blockAblation": round(abl_block[l], 4),
            }
            for l in range(layers)
        ],
        "finding": (
            "Attribution and necessity agree only weakly at this scale: the "
            "rank correlations are positive but modest, and the top-five layer "
            "sets overlap for only one attention layer. An attribution peak is "
            "therefore a hypothesis about where to intervene, not a circuit."
        ),
    }


def build_acdc():
    graph = read("graph")
    prompts = []
    curves = []
    kept_sets = []
    retentions = []

    for pid, entry in graph.items():
        base = readout(entry, entry["edges"])
        kept = list(entry["edges"])
        removed = []
        by_budget = {}
        while len(kept) > BUDGET:
            best = None
            for i, edge in enumerate(kept):
                trial = kept[:i] + kept[i + 1:]
                value = readout(entry, trial)
                score = abs(value - base)
                if best is None or score < best[0]:
                    best = (score, i, value)
            assert best is not None
            _, i, _ = best
            removed.append(kept.pop(i))
            if len(kept) in (128, 112, 96, 80, 64):
                by_budget[str(len(kept))] = round(readout(entry, kept) / base, 4)

        final = readout(entry, kept)
        retention = final / base
        kept_sets.append({(e["s"], e["t"]) for e in kept})
        retentions.append(retention)
        prompts.append({
            "id": pid,
            "baseline": round(base, 3),
            "final": round(final, 3),
            "retention": round(retention, 4),
            "candidates": len(entry["edges"]),
            "kept": len(kept),
            "chainKept": sum(1 for e in kept if e["kind"] == "chain"),
            "attnKept": sum(1 for e in kept if e["kind"] == "attn"),
            "removed": len(removed),
        })
        curves.append({"id": pid, **by_budget, "final": round(retention, 4)})

    stable = set.intersection(*kept_sets) if kept_sets else set()

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "offline ACDC-style greedy edge pruning on the captured wiring graph, using a linear readout proxy",
        "causal": False,
        "budget": BUDGET,
        "candidateEdges": sorted({len(v["edges"]) for v in graph.values()}),
        "prompts": prompts,
        "curves": curves,
        "aggregate": {
            "meanRetention": round(mean(retentions), 4),
            "minRetention": round(min(retentions), 4),
            "maxRetention": round(max(retentions), 4),
            "stableEdges": len(stable),
        },
        "caveat": (
            "This pilot propagates stored direct attributions over the captured "
            "wiring graph; it does not re-run the model or ablate a model edge. "
            "The retention numbers are a test of the graph arithmetic, not a "
            "causal ACDC circuit. A real ACDC run still needs mean-ablation "
            "forward passes, a reference distribution and held-out prompts."
        ),
    }


def readout(entry, edges):
    """Linear propagation of stored target attribution through the wiring graph.

    Token nodes start at 1. A component receives the weighted sum of its incoming
    edges. The metric is the sum of absolute readout mass over attention/MLP
    components. This is a graph surrogate, not a model logit.
    """
    incoming = {n["id"]: [] for n in entry["nodes"]}
    for edge in edges:
        incoming[edge["t"]].append(edge)
    kind_order = {"token": 0, "attn": 1, "mlp": 2}
    order = sorted(entry["nodes"], key=lambda n: (n["layer"], kind_order[n["kind"]]))
    value = {}
    for node in order:
        if node["kind"] == "token":
            value[node["id"]] = 1.0
        else:
            value[node["id"]] = sum(
                value.get(edge["s"], 0.0) * float(edge["w"])
                for edge in incoming[node["id"]]
            )
    return sum(
        abs(value[n["id"]] * float(n.get("attr", 0.0)))
        for n in entry["nodes"] if n["kind"] != "token"
    )


def write(name, obj):
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.1f} kB)")


def main():
    mechanism = build_mechanism()
    acdc = build_acdc()
    write("mechanism", mechanism)
    write("acdc", acdc)
    print("mechanism agreement:", mechanism["agreement"])
    print("acdc aggregate:", acdc["aggregate"])


if __name__ == "__main__":
    main()
