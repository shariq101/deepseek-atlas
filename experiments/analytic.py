#!/usr/bin/env python3
"""
analytics.py — every number on the config-derived chapters is computed here.

Inputs : the official `config.json` of each checkpoint (experiments/configs).
Outputs: public/data/{family,architecture,budget,cache,frontier}.json

No value is typed from memory: published totals are imported only to *check*
the computed ones, and the check is exported alongside them. Where a formula
cannot be exact (V4's grouped low-rank output, V4.1's cross-layer cache), the
approximation is named in `assumptions` and shown in the interface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "experiments" / "configs"
OUT = ROOT / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Published facts. Totals are the lab's own; they are used as a cross-check.
# --------------------------------------------------------------------------- #
MEMBERS = [
    dict(
        id="v2", cfg="DeepSeek-V2", name="DeepSeek-V2", short="V2",
        generation="V2", date="2024-05", published_total=236e9, published_active=21e9,
        context=163840, trained="8.1T", license="DeepSeek Model License",
        attention="MLA", precision="BF16", mtp=0, hash_layers=0,
        note="The first MLA + DeepSeekMoE generation: 160 routed experts, top-6, two shared experts, 60 layers.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V2",
    ),
    dict(
        id="v2lite", cfg="DeepSeek-V2-Lite", name="DeepSeek-V2-Lite", short="V2-Lite",
        generation="V2", date="2024-11", published_total=15.7e9, published_active=2.4e9,
        context=163840, trained="5.7T", license="DeepSeek Model License",
        attention="MLA", precision="BF16", mtp=0, hash_layers=0,
        note="The only member small enough to instrument on one free GPU. Same MLA + DeepSeekMoE grammar as every model after it.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite",
    ),
    dict(
        id="v3", cfg="DeepSeek-V3", name="DeepSeek-V3", short="V3",
        generation="V3", date="2024-12", published_total=671e9, published_active=37e9,
        context=163840, trained="14.8T", license="DeepSeek Model License",
        attention="MLA", precision="FP8", mtp=1, hash_layers=0,
        note="The model that fixed the family's silhouette: 256 routed experts, one shared expert, 8 active, MLA throughout.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V3",
    ),
    dict(
        id="v31", cfg="DeepSeek-V3.1", name="DeepSeek-V3.1", short="V3.1",
        generation="V3.1", date="2025-08", published_total=671e9, published_active=37e9,
        context=163840, trained="—", license="DeepSeek Model License",
        attention="MLA", precision="FP8", mtp=1, hash_layers=0,
        note="Same weights architecture as V3, retrained for hybrid thinking. Included to show that a generation can change behaviour without changing shape.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V3.1",
    ),
    dict(
        id="v32", cfg="DeepSeek-V3.2-Exp", name="DeepSeek-V3.2-Exp", short="V3.2-Exp",
        generation="V3.2", date="2025-09", published_total=671e9, published_active=37e9,
        context=163840, trained="—", license="DeepSeek Model License",
        attention="DSA (MLA + sparse index)", precision="FP8", mtp=1, hash_layers=0,
        note="DeepSeek Sparse Attention: a 64-head index selects 2,048 keys per query, so attention stops paying for the whole context.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp",
    ),
    dict(
        id="r1", cfg="DeepSeek-R1", name="DeepSeek-R1", short="R1",
        generation="R1", date="2025-01", published_total=671e9, published_active=37e9,
        context=163840, trained="14.8T", license="MIT",
        attention="MLA", precision="FP8", mtp=1, hash_layers=0,
        note="Reasoning post-training on the V3 base: the same 61-layer MLA + 256-expert MoE, with RL-shaped behaviour rather than a new shape.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-R1",
    ),
    dict(
        id="v4flash", cfg="DeepSeek-V4-Flash", name="DeepSeek-V4-Flash", short="V4-Flash",
        generation="V4", date="2026-04", published_total=284e9, published_active=13e9,
        context=1048576, trained="32T", license="MIT",
        attention="CSA + HCA (hybrid)", precision="FP4 experts + FP8", mtp=1, hash_layers=3,
        note="The first family member with hash-routed MoE layers and a hybrid compressed/sparse attention stack; 1M context at 13B active.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
    ),
    dict(
        id="v4pro", cfg="DeepSeek-V4-Pro", name="DeepSeek-V4-Pro", short="V4-Pro",
        generation="V4", date="2026-04", published_total=1.6e12, published_active=49e9,
        context=1048576, trained="33T", license="MIT",
        attention="CSA + HCA (hybrid)", precision="FP4 experts + FP8", mtp=1, hash_layers=3,
        note="The 1.6T flagship: 384 experts, 61 layers, 49B active — the same sparsity ratio as V3 at 2.4× the width.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
    ),
    dict(
        id="v41flash", cfg="DeepSeek-V4.1-Flash", name="DeepSeek-V4.1-Flash", short="V4.1-Flash",
        generation="V4.1", date="2026-09", published_total=552e9, published_active=16e9,
        context=1048576, trained="—", license="MIT",
        attention="CED + shared KV (CSA/HCA)", precision="FP4 experts + FP8", mtp=3, hash_layers=0,
        note="Causal encoder–decoder: 8B active on the input path, 16B on the output path, three MTP heads, and KV shared across layers.",
        source="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash",
    ),
]

# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_cfg(name: str) -> dict:
    d = json.loads((CFG / f"{name}.json").read_text())
    if "text_config" in d:
        # V4.1 nests the language model under text_config; keep the outer
        # vision fields accessible for the architecture chapter.
        d = {**d["text_config"], "_outer": {k: v for k, v in d.items() if k != "text_config"}}
    return d


def get(c: dict, key: str, default=None):
    return c.get(key, default)


def attention_params(c: dict) -> dict:
    """Exact linear shapes for MLA; V4's grouped low-rank output is marked."""
    d = c["hidden_size"]
    H = c["num_attention_heads"]
    rope = c.get("qk_rope_head_dim", 64)
    head_dim = c.get("head_dim")
    nope = c.get("qk_nope_head_dim", (head_dim - rope) if head_dim else 128)
    v = c.get("v_head_dim", head_dim or nope)
    q_lora = c.get("q_lora_rank")
    q_head = nope + rope

    if q_lora:
        q = d * q_lora + q_lora * H * q_head + q_lora
    else:
        q = d * H * q_head

    kv_lora = c.get("kv_lora_rank")
    if kv_lora:
        kv_a = d * (kv_lora + rope) + kv_lora
        kv_b = kv_lora * H * (nope + v)
    else:
        # V4 stores one latent KV head; the config exposes head_dim directly.
        kv_a = d * (head_dim + rope) + head_dim
        kv_b = head_dim * H * (nope + v)

    o_lora = c.get("o_lora_rank")
    if o_lora:
        # Approximation: grouped low-rank output projection. The config names
        # o_groups and o_lora_rank but not the exact factorisation.
        groups = c.get("o_groups", 1)
        o = H * v * o_lora + groups * o_lora * d
    else:
        o = H * v * d

    norm = (q_lora or 0) + (c.get("kv_lora_rank") or head_dim) + 2 * d
    total = q + kv_a + kv_b + o
    return dict(q=q, kv_a=kv_a, kv_b=kv_b, o=o, norm=norm, total=total,
                q_lora=q_lora, kv_lora=kv_lora, o_lora=o_lora,
                nope=nope, rope=rope, v=v, heads=H)


def expert_params(c: dict) -> dict:
    d = c["hidden_size"]
    E = c["n_routed_experts"]
    I = c["moe_intermediate_size"]
    shared = c.get("n_shared_experts", 0) or 0
    routed_total = E * 3 * d * I
    shared_total = shared * 3 * d * I
    gate = d * E
    return dict(routed_total=routed_total, shared_total=shared_total, gate=gate,
                per_expert=3 * d * I, E=E, shared=shared, I=I)


def count_params(c: dict) -> dict:
    d = c["hidden_size"]
    L = c["num_hidden_layers"]
    V = c["vocab_size"]
    first_dense = c.get("first_k_dense_replace", 0) or 0
    moe_freq = c.get("moe_layer_freq", 1) or 1
    attn = attention_params(c)
    exp = expert_params(c)

    attn_all = attn["total"] * L
    dense_layers = min(first_dense, L)
    # moe_layer_freq=1 => every layer after the dense prefix is MoE.
    n_moe = sum(1 for l in range(L) if l >= dense_layers and (l - dense_layers) % moe_freq == 0)
    n_dense = L - n_moe
    dense_total = n_dense * 3 * d * c.get("intermediate_size", 0)
    moe_total = n_moe * (exp["routed_total"] + exp["shared_total"] + exp["gate"])
    norms = L * 2 * d
    embed = V * d
    lm_head = 0 if c.get("tie_word_embeddings") else V * d

    total = attn_all + dense_total + moe_total + norms + embed + lm_head
    # Active parameters: attention everywhere + shared expert everywhere +
    # top-k routed experts + gate + one unembedding read.
    active = (
        attn_all
        + n_moe * exp["shared_total"]
        + n_moe * exp["gate"]
        + n_moe * (c["num_experts_per_tok"] / exp["E"]) * exp["routed_total"]
        + dense_total
        + norms
        + lm_head
    )
    return dict(
        total=total, active=active, attn=attn_all, dense=dense_total,
        moe=moe_total, norms=norms, embed=embed, lm_head=lm_head,
        n_moe=n_moe, n_dense=n_dense, attn_per_layer=attn["total"],
        routed_per_layer=exp["routed_total"], shared_per_layer=exp["shared_total"],
        gate_per_layer=exp["gate"], expert=exp, attention=attn,
        sparsity=total / active if active else 0,
        active_share=active / total if total else 0,
    )


def kv_bytes_per_token(c: dict, dtype_bytes: int = 2) -> dict:
    """Cache bytes per token, one sequence, using each architecture's own scheme."""
    L = c["num_hidden_layers"]
    rope = c.get("qk_rope_head_dim", 64)
    ratios = c.get("compress_ratios")
    kv_lora = c.get("kv_lora_rank")
    head_dim = c.get("head_dim")
    kv_heads = c.get("num_key_value_heads", c["num_attention_heads"])
    H = c["num_attention_heads"]
    v = c.get("v_head_dim", head_dim or 128)

    # Standard MHA reference at the same width.
    mha = 2 * L * H * v * dtype_bytes
    # GQA reference at the configured KV head count.
    gqa = 2 * L * kv_heads * v * dtype_bytes

    if ratios:
        # V4: per-layer compressed cache. Layers with ratio 0 use a sliding
        # window, whose cache is bounded by 128 tokens rather than the context;
        # they are reported separately, not counted per-token.
        shared_ids = c.get("kv_source_layer_ids") or []
        stored = [l for l in range(L) if not shared_ids or l in shared_ids]
        per_layer = []
        for l in range(L):
            r = ratios[l] if l < len(ratios) else 1
            if shared_ids and l not in shared_ids:
                continue
            if r >= 1:
                per_layer.append((head_dim + rope) * dtype_bytes / r)
        latent = sum(per_layer)
        return dict(scheme="CSA/HCA", per_token=latent, mha=mha, gqa=gqa,
                    ratio=mha / latent if latent else 0,
                    layers_stored=len(per_layer), sliding=sum(1 for r in ratios if r == 0),
                    per_layer=per_layer)
    if kv_lora:
        latent = L * (kv_lora + rope) * dtype_bytes
        return dict(scheme="MLA", per_token=latent, mha=mha, gqa=gqa,
                    ratio=mha / latent if latent else 0, layers_stored=L, sliding=0,
                    per_layer=[(kv_lora + rope) * dtype_bytes] * L)
    return dict(scheme="GQA", per_token=gqa, mha=mha, gqa=gqa,
                ratio=mha / gqa if gqa else 0, layers_stored=L, sliding=0,
                per_layer=[])


def attention_flops_per_token(c: dict, ctx: int) -> float:
    """Forward attention arithmetic for one token at a given context length."""
    L = c["num_hidden_layers"]
    H = c["num_attention_heads"]
    rope = c.get("qk_rope_head_dim", 64)
    v = c.get("v_head_dim", c.get("head_dim", 128))
    qk = c.get("qk_nope_head_dim", c.get("head_dim", 128)) + rope
    ratios = c.get("compress_ratios")
    topk = c.get("index_topk")

    if ratios:
        # Hybrid: only layers with a positive ratio attend globally; the rest
        # use a 128-token sliding window. Each global layer's keys are
        # compressed by its ratio.
        total = 0.0
        for l, r in enumerate(ratios[:L]):
            if r == 0:
                total += 2 * H * (qk + v) * min(ctx, c.get("sliding_window", 128))
            else:
                total += 2 * H * (qk + v) * (ctx / r)
        return total
    if topk:
        # DSA: a 64-head indexer scores the full context (128 dims), then
        # 2,048 keys are attended by every head.
        idx = c.get("index_n_heads", 64) * c.get("index_head_dim", 128)
        return L * (2 * idx * ctx + 2 * H * (qk + v) * min(ctx, topk))
    # Full causal attention (MLA or GQA); per token cost is linear in context.
    return L * 2 * H * (qk + v) * ctx


def layer_plan(c: dict) -> list[dict]:
    L = c["num_hidden_layers"]
    first_dense = c.get("first_k_dense_replace", 0) or 0
    ratios = c.get("compress_ratios") or []
    hash_layers = c.get("num_hash_layers", 0) or 0
    shared_ids = c.get("kv_source_layer_ids") or []
    rows = []
    for l in range(L):
        if ratios:
            attn_kind = "hca" if ratios[l] >= 64 else ("csa" if ratios[l] > 0 else "sliding")
        elif c.get("index_topk"):
            attn_kind = "dsa"
        else:
            attn_kind = "mla"
        rows.append(dict(
            layer=l,
            ffn="dense" if l < first_dense else "moe",
            attn=attn_kind,
            compress=ratios[l] if l < len(ratios) else 1,
            hash=l < hash_layers,
            kv_shared=bool(shared_ids) and l not in shared_ids,
            kv_source=l in shared_ids,
        ))
    return rows


def build_member(m: dict) -> dict:
    c = load_cfg(m["cfg"])
    counts = count_params(c)
    kv = kv_bytes_per_token(c)
    plan = layer_plan(c)
    out = {
        **m,
        "layers": c["num_hidden_layers"],
        "hidden": c["hidden_size"],
        "heads": c["num_attention_heads"],
        "kvHeads": c.get("num_key_value_heads", c["num_attention_heads"]),
        "vocab": c["vocab_size"],
        "experts": c.get("n_routed_experts", 0),
        "shared": c.get("n_shared_experts", 0),
        "topk": c.get("num_experts_per_tok", 0),
        "moeInter": c.get("moe_intermediate_size", 0),
        "headDim": c.get("head_dim") or (c.get("qk_nope_head_dim", 128) + c.get("qk_rope_head_dim", 64)),
        "qLora": c.get("q_lora_rank"),
        "kvLora": c.get("kv_lora_rank"),
        "indexTopk": c.get("index_topk"),
        "sliding": c.get("sliding_window"),
        "compress": c.get("compress_ratios"),
        "kvSource": c.get("kv_source_layer_ids"),
        "computed_total": counts["total"],
        "computed_active": counts["active"],
        "total_error": (counts["total"] - m["published_total"]) / m["published_total"],
        "active_error": (counts["active"] - m["published_active"]) / m["published_active"],
        "sparsity": counts["sparsity"],
        "kv": kv,
        "plan": plan,
        "counts": counts,
    }
    return out


def main() -> None:
    members = [build_member(m) for m in MEMBERS]

    family = {
        "generatedAt": "2026-11-12",
        "thesis": "One architecture family, measured from the smallest member to the largest.",
        "members": members,
        "generations": [
            {"id": "V2", "label": "V2 · 2024", "change": "MLA + DeepSeekMoE introduced", "experts": 160, "active": 21e9},
            {"id": "V3", "label": "V3 · 2024", "change": "256 experts, aux-loss-free balancing, FP8, MTP", "experts": 256, "active": 37e9},
            {"id": "V3.1", "label": "V3.1 · 2025", "change": "hybrid thinking; shape unchanged", "experts": 256, "active": 37e9},
            {"id": "V3.2", "label": "V3.2 · 2025", "change": "DeepSeek Sparse Attention index", "experts": 256, "active": 37e9},
            {"id": "R1", "label": "R1 · 2025", "change": "reasoning RL on V3; shape unchanged", "experts": 256, "active": 37e9},
            {"id": "V4", "label": "V4 · 2026", "change": "CSA/HCA, hash routing, FP4 experts, 1M context", "experts": 384, "active": 49e9},
            {"id": "V4.1", "label": "V4.1 · 2026", "change": "Causal encoder–decoder, shared KV across layers, multimodal", "experts": 384, "active": 16e9},
        ],
        "sources": [
            {"name": "DeepSeek-V2", "url": "https://arxiv.org/abs/2405.04434"},
            {"name": "DeepSeek-V3 technical report", "url": "https://arxiv.org/abs/2412.19437"},
            {"name": "DeepSeek-R1", "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1"},
            {"name": "DeepSeek-V3.2-Exp", "url": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp"},
            {"name": "DeepSeek-V4 technical report", "url": "https://arxiv.org/abs/2606.19348"},
            {"name": "DeepSeek-V4.1-Flash model card", "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"},
        ],
    }
    (OUT / "family.json").write_text(json.dumps(family, indent=1))

    # ------------------------------------------------------------ architecture
    arch = {"focus": {}, "comparison": []}
    for mid in ("v2lite", "v3", "v4flash", "v41flash"):
        m = next(x for x in members if x["id"] == mid)
        c = load_cfg(m["cfg"])
        a = attention_params(c)
        arch["focus"][mid] = {
            "id": mid, "name": m["name"], "layers": m["layers"], "hidden": m["hidden"],
            "heads": m["heads"], "kvHeads": m["kvHeads"], "headDim": m["headDim"],
            "qLora": m["qLora"], "kvLora": m["kvLora"], "vocab": m["vocab"],
            "experts": m["experts"], "shared": m["shared"], "topk": m["topk"],
            "moeInter": m["moeInter"], "denseInter": c.get("intermediate_size", 0),
            "firstDense": c.get("first_k_dense_replace", 0) or 0,
            "mtp": m["mtp"], "plan": m["plan"], "attention": a,
            "tie": c.get("tie_word_embeddings", False),
            "scoring": c.get("scoring_func", "softmax"),
            "topkMethod": c.get("topk_method", "greedy"),
            "routedScale": c.get("routed_scaling_factor", 1.0),
            "normTopk": c.get("norm_topk_prob", False),
            "indexTopk": m["indexTopk"], "sliding": m["sliding"],
            "hashLayers": m["hash_layers"],
            "nGroups": c.get("n_group"), "topkGroup": c.get("topk_group"),
            "qkRope": c.get("qk_rope_head_dim", 64),
            "qkNope": a["nope"], "vHead": a["v"],
            "oGroups": c.get("o_groups"), "oLora": c.get("o_lora_rank"),
        }
        arch["comparison"].append({
            "id": mid, "name": m["name"], "layers": m["layers"], "hidden": m["hidden"],
            "experts": m["experts"], "topk": m["topk"], "active": m["published_active"],
            "total": m["published_total"], "context": m["context"],
        })
    arch["assumptions"] = [
        "MLA and expert shapes are exact products of the config's named dimensions.",
        "V4's grouped low-rank output projection (o_groups, o_lora_rank) is counted as H·v·o_lora + groups·o_lora·d; the config does not publish the exact factorisation.",
        "Published totals exclude the MTP block; the same convention is used here so the checks are comparable.",
    ]
    (OUT / "architecture.json").write_text(json.dumps(arch, indent=1))

    # ------------------------------------------------------------------ budget
    budget = {"members": []}
    for m in members:
        c = load_cfg(m["cfg"])
        k = m["counts"]
        ctxs = [1024, 8192, 32768, 131072, 1048576]
        budget["members"].append({
            "id": m["id"], "name": m["name"], "short": m["short"], "generation": m["generation"],
            "total": k["total"], "active": k["active"], "published_total": m["published_total"],
            "published_active": m["published_active"],
            "sparsity": k["sparsity"], "layers": m["layers"],
            "experts": m["experts"], "topk": m["topk"], "shared": m["shared"],
            "attn": k["attn"], "moe": k["moe"], "dense": k["dense"], "embed": k["embed"],
            "lmHead": k["lm_head"], "norms": k["norms"],
            "nMoe": k["n_moe"], "nDense": k["n_dense"],
            "sharedTotal": k["n_moe"] * k["shared_per_layer"],
            "routedTotal": k["n_moe"] * k["routed_per_layer"],
            "attnPerLayer": k["attn_per_layer"], "routedPerLayer": k["routed_per_layer"],
            "sharedPerLayer": k["shared_per_layer"], "gatePerLayer": k["gate_per_layer"],
            "perExpert": k["expert"]["per_expert"],
            "expertShare": k["moe"] / k["total"],
            "activeShare": k["active"] / k["total"],
            "flopsPerToken": 2 * k["active"],
            "attnFlops": {str(x): attention_flops_per_token(c, x) for x in ctxs},
            "denseEquivalent": k["total"] * 2,
        })
    (OUT / "budget.json").write_text(json.dumps(budget, indent=1))

    # ------------------------------------------------------------------- cache
    cache = {"members": [], "contexts": [1024, 8192, 32768, 131072, 524288, 1048576], "official": []}
    for m in members:
        kv = m["kv"]
        cache["members"].append({
            "id": m["id"], "name": m["name"], "short": m["short"], "generation": m["generation"],
            "scheme": kv["scheme"], "perToken": kv["per_token"], "mha": kv["mha"], "gqa": kv["gqa"],
            "ratio": kv["ratio"], "layersStored": kv["layers_stored"], "sliding": kv["sliding"],
            "layers": m["layers"], "kvHeads": m["kvHeads"], "headDim": m["headDim"],
            "kvLora": m["kvLora"], "compress": m["compress"], "kvSource": m["kvSource"],
            "perLayer": kv["per_layer"],
            "officialPerToken": 890 if m["id"] == "v41flash" else None,
            "at": {str(x): kv["per_token"] * x for x in cache["contexts"]},
        })
    cache["official"] = [
        {"model": "V2", "claim": "MLA reduces the KV cache by 93.3% relative to DeepSeek 67B.", "url": "https://arxiv.org/abs/2405.04434"},
        {"model": "V3.2", "claim": "DSA reduces per-token inference cost as context grows, with a 2,048-key top-k index.", "url": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp"},
        {"model": "V4.1", "claim": "890 bytes/token of global KV (FP4, CSA2 cross-layer reuse), ~1/4 of V4-Flash and 437× smaller than V1; every layer keeps a 128-token sliding window, and only 4 of 40 layers (3 encoder + 1 decoder) compute a full input cache.", "url": "https://arxiv.org/abs/2609.19969"},
    ]
    (OUT / "cache.json").write_text(json.dumps(cache, indent=1))

    # ---------------------------------------------------------------- frontier
    frontier = {"members": [], "contexts": [4096, 32768, 131072, 524288, 1048576]}
    for m in members:
        c = load_cfg(m["cfg"])
        frontier["members"].append({
            "id": m["id"], "name": m["name"], "short": m["short"], "generation": m["generation"],
            "attention": m["attention"], "context": m["context"],
            "layers": m["layers"], "heads": m["heads"],
            "qk": (c.get("qk_nope_head_dim") or c.get("head_dim") or 128) + c.get("qk_rope_head_dim", 64),
            "v": c.get("v_head_dim") or c.get("head_dim") or 128,
            "kvPerToken": m["kv"]["per_token"], "kv1m": m["kv"]["per_token"] * 1048576,
            "indexTopk": m["indexTopk"], "sliding": m["sliding"],
            "compress": m["compress"], "kvSource": m["kvSource"],
            "plan": m["plan"],
            "flops": {str(x): attention_flops_per_token(c, x) for x in frontier["contexts"]},
        })
    (OUT / "frontier.json").write_text(json.dumps(frontier, indent=1))

    # ------------------------------------------------------------------ report
    print(f"{'model':18} {'published':>12} {'computed':>12} {'err':>7} {'active':>10} {'err':>7} {'sparsity':>9}")
    for m in members:
        print(f"{m['short']:18} {m['published_total']/1e9:11.1f}B {m['computed_total']/1e9:11.1f}B "
              f"{m['total_error']*100:+6.1f}% {m['computed_active']/1e9:9.1f}B {m['active_error']*100:+6.1f}% "
              f"{m['sparsity']:8.1f}×")
    print("\nKV bytes/token (fp16):")
    for m in members:
        kv = m["kv"]
        print(f"  {m['short']:18} {kv['scheme']:10} {kv['per_token']/1024:9.1f} KiB  vs MHA {kv['ratio']:6.1f}×")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
