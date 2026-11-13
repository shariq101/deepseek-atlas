#!/usr/bin/env python3
"""
suite.py — capture a DeepSeek-family model into the atlas' JSON artefacts.

Designed for one free Kaggle GPU and a 15.7B checkpoint:

    MODEL_ID=deepseek-ai/DeepSeek-V2-Lite QUANT=4bit python suite.py all

Stages are cached under $ATLAS_CACHE (default /kaggle/working/atlas_cache), so a
disconnected kernel resumes rather than restarts. The model is loaded once and
every instrument is a forward pass over the same 27-prompt contrast corpus.
Everything written here is measured; nothing is sampled or illustrated.

Outputs (in $ATLAS_OUT, default ./data):
  meta.json prompts.json lens.json attention.json attention_full.json
  router.json experts.json norms.json dla.json ablation.json probes.json
  geometry.json graph.json context.json
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:  # when concatenated into the Kaggle kernel, corpus is defined above
    from corpus import (
        PROMPTS, DEEP, PROBES, PROBE_PROPS, NEEDLE, NEEDLE_Q, NEEDLE_TARGET,
        NEEDLE_LENGTHS, NEEDLE_DEPTHS, FILLER,
    )
except ImportError:
    pass

os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf")
os.environ.setdefault("ATLAS_OUT", "/kaggle/working/data")
os.environ.setdefault("ATLAS_CACHE", "/kaggle/tmp/atlas_cache")

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-ai/DeepSeek-V2-Lite")
QUANT = os.environ.get("QUANT", "4bit")
DTYPE = os.environ.get("DTYPE", "float16")
LIMIT = int(os.environ.get("LIMIT", "0"))
STAGES = os.environ.get("STAGES", "all")
CACHE = Path(os.environ.get("ATLAS_CACHE", "/kaggle/working/atlas_cache"))
OUT = Path(os.environ.get("ATLAS_OUT", "data"))
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ABL_PROMPTS = ["fact-capital", "fact-author", "syn-attract", "syn-plural", "ind-1",
               "arith-add", "arith-mul", "code-add", "multi-fr", "neg-always",
               "idiom-bucket", "ret-code"]
ABL_EXPERTS = [0, 7, 15, 23, 31, 39, 47, 55]
ABL_LAYERS = [3, 7, 11, 15, 19, 23]

# Causal follow-up. These are deliberately smaller than the full candidate graph:
# four prompts for head/expert ablation, one discovery prompt and one held-out
# prompt for the residual-write edge search. Every number is a model forward pass.
CAUSAL_PROMPTS = ["fact-capital", "syn-attract", "ind-1", "code-add"]
EDGE_DISCOVERY = "fact-capital"
EDGE_VALIDATION = "syn-attract"
EDGE_BUDGET = 24


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[atlas {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def r4(x, n=4):
    """Round for JSON without losing sign or producing -0.0."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    if isinstance(x, np.ndarray):
        return [r4(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [r4(v, n) for v in x]
    v = float(x)
    if not math.isfinite(v):
        return 0.0
    return round(v, n) + 0.0


def save(name: str, obj) -> None:
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, separators=(",", ":")))
    log(f"wrote {p.name} ({p.stat().st_size/1e6:.1f} MB)")


def stage_done(name: str) -> bool:
    return (CACHE / f"{name}.done").exists()


def mark_done(name: str) -> None:
    (CACHE / f"{name}.done").write_text(str(time.time()))


def cached(name: str):
    p = OUT / f"{name}.json"
    if stage_done(name) and p.exists():
        data = json.loads(p.read_text())
        if data:
            log(f"stage '{name}' cached")
            return data
        log(f"cached stage '{name}' is empty; recomputing")
    return None


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
# A stage can finish without raising and still write an empty object. The old
# capture did exactly that: meta.json said 0 prompts while lens.json and
# router.json were {}. These checks are intentionally structural: they cannot
# validate the science, but they stop an empty run from being shipped as data.
STAGE_OUTPUTS = {
    "capture": ["meta", "prompts", "lens", "attention", "router", "norms", "dla", "latent_raw"],
    "ablation": ["ablation"],
    "probes": ["probes"],
    "geometry": ["geometry"],
    "latent": ["latent"],
    "context": ["context"],
    "graph": ["graph"],
    "experts": ["experts"],
    "interventions": ["interventions"],
    "edges": ["edges"],
}

REQUIRED_KEYS = {
    "meta": ("modelId", "layers", "corpus"),
    "ablation": ("layers", "prompts"),
    "probes": ("properties", "byProp"),
    "geometry": ("cka", "tokenTrajectories"),
    "latent": ("norms", "pca"),
    "context": ("results",),
    "graph": (),
    "experts": ("load", "coactivation", "profiles"),
}


def _empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (dict, list, tuple, str)):
        return len(v) == 0
    return False


def _check_artifact(name: str, obj) -> list[str]:
    errors: list[str] = []
    if _empty(obj):
        return [f"{name}.json is empty"]
    if name == "prompts":
        if not isinstance(obj, list):
            return ["prompts.json must be a list"]
        if not all(isinstance(p, dict) and p.get("id") for p in obj):
            errors.append("prompts.json contains an entry without an id")
        return errors
    if not isinstance(obj, dict):
        return [f"{name}.json must be an object"]
    for key in REQUIRED_KEYS.get(name, ()):
        if key not in obj or _empty(obj[key]):
            errors.append(f"{name}.json has an empty/missing field '{key}'")
    if name == "meta":
        if int(obj.get("layers") or 0) < 1:
            errors.append("meta.json reports zero layers")
        if int(obj.get("corpus") or 0) < 1:
            errors.append("meta.json reports zero captured prompts")
    return errors


def preflight(stages: list[str], out: Path | None = None) -> list[str]:
    """Validate the outputs of the stages that were requested.

    Returns a list of human-readable errors. An empty list means every requested
    stage produced a non-empty artefact with the fields the frontend consumes.
    """
    out = out or OUT
    errors: list[str] = []
    for stage in stages:
        if stage not in STAGE_OUTPUTS:
            errors.append(f"unknown stage '{stage}'")
            continue
        for name in STAGE_OUTPUTS[stage]:
            p = out / f"{name}.json"
            if not p.exists():
                errors.append(f"missing {name}.json (stage {stage})")
                continue
            try:
                obj = json.loads(p.read_text())
            except Exception as exc:
                errors.append(f"unreadable {name}.json: {exc}")
                continue
            errors.extend(_check_artifact(name, obj))
    if errors:
        log(f"PREFLIGHT FAILED ({len(errors)} issue{'s' if len(errors) != 1 else ''})")
        for err in errors:
            log(f"  - {err}")
    else:
        log(f"preflight ok: {', '.join(stages)}")
    return errors


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
model = None
tok = None
W_U = None          # dequantised unembedding, float16 on device
CURRENT = {}        # per-forward hook capture
CAPTURE_ON = False


def load_model():
    global model, tok, W_U
    from transformers import AutoTokenizer, AutoModelForCausalLM

    import shutil
    try:
        du = shutil.disk_usage("/kaggle/tmp")
        log(f"scratch {du.free/2**30:.0f} GiB free · working {shutil.disk_usage('/kaggle/working').free/2**30:.0f} GiB free")
    except Exception:
        pass
    log(f"loading tokenizer {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    kw = dict(trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True)
    if QUANT == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    else:
        kw["torch_dtype"] = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
    try:
        kw["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
    except TypeError:
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()

    model.config.use_cache = False
    w = model.lm_head.weight
    if hasattr(w, "dequantize"):
        try:
            w = w.dequantize()
        except Exception:
            w = w.data
    W_U = w.detach().to("cuda" if torch.cuda.is_available() else "cpu").half()

    if torch.cuda.is_available():
        log(f"gpu {torch.cuda.get_device_name(0)} · allocated {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    log(f"model ready: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters "
        f"({QUANT})")


def cfg_int(key, default=None):
    return getattr(model.config, key, default)


def hidden_dtype():
    return next(model.parameters()).dtype


def register_hooks():
    """One set of forward hooks, always installed; CAPTURE_ON gates writing."""
    for i, layer in enumerate(model.model.layers):
        def attn_hook(m, inp, out, i=i):
            if not CAPTURE_ON:
                return
            h = out[0] if isinstance(out, tuple) else out
            CURRENT.setdefault("attn", {})[i] = h.detach()
        layer.self_attn.register_forward_hook(attn_hook)

        def mlp_hook(m, inp, out, i=i):
            if not CAPTURE_ON:
                return
            h = out[0] if isinstance(out, tuple) else out
            CURRENT.setdefault("mlp", {})[i] = h.detach()
        layer.mlp.register_forward_hook(mlp_hook)

        mlp = layer.mlp
        if hasattr(mlp, "gate"):
            def gate_hook(m, inp, out, i=i):
                if not CAPTURE_ON:
                    return
                hs = inp[0]
                logits = F.linear(hs.float(), m.weight.float())
                if getattr(model.config, "scoring_func", "softmax") == "softmax":
                    scores = logits.softmax(-1)
                else:
                    scores = logits.sigmoid()
                idx, wgt = out[0], out[1]
                CURRENT.setdefault("gate", {})[i] = (scores.detach(), idx, wgt)
            mlp.gate.register_forward_hook(gate_hook)
        sa = layer.self_attn
        if hasattr(sa, "kv_a_proj_with_mqa"):
            def kv_hook(m, inp, out, i=i):
                if not CAPTURE_ON:
                    return
                CURRENT.setdefault("kv", {})[i] = out.detach()
            sa.kv_a_proj_with_mqa.register_forward_hook(kv_hook)
        if hasattr(sa, "q_a_proj"):
            def q_hook(m, inp, out, i=i):
                if not CAPTURE_ON:
                    return
                CURRENT.setdefault("q", {})[i] = out.detach()
            sa.q_a_proj.register_forward_hook(q_hook)
        if hasattr(mlp, "shared_experts"):
            def shared_hook(m, inp, out, i=i):
                if not CAPTURE_ON:
                    return
                CURRENT.setdefault("shared", {})[i] = out.detach()
            mlp.shared_experts.register_forward_hook(shared_hook)


def n_layers():
    return len(model.model.layers)


def norm_hidden(h, apply_norm=True):
    """Final RMSNorm. hidden_states[L] comes out of the model already normed."""
    if not apply_norm:
        return h
    if h.dim() == 1:
        h = h.unsqueeze(0)
    norm = model.model.norm
    # device_map="auto" can place the final norm and lm_head on cuda:0 while
    # later hidden layers run on cuda:1. Move the activation to the norm.
    try:
        dev = next(norm.parameters()).device
    except StopIteration:
        dev = h.device
    return norm(h.to(device=dev, dtype=hidden_dtype()))


def state_at(hs, layer, pos):
    """One token's hidden vector, whether hidden_states is [B,T,H] or [T,H].

    HuggingFace normally returns [B,T,H], but a 2D [T,H] tuple is enough to
    turn hs[l][0, q] into a scalar. That scalar is what reached F.linear as a
    0D input in the failed Kaggle capture.
    """
    t = hs[layer]
    if t.dim() == 3:
        return t[0, pos]
    if t.dim() == 2:
        return t[pos]
    if t.dim() == 1:
        return t
    raise ValueError(f"unexpected hidden-state shape {tuple(t.shape)} at layer {layer}")


def seq_at(hs, layer):
    t = hs[layer]
    return t[0] if t.dim() == 3 else t


def lens_all(h, apply_norm=True):
    """Full logit-lens distribution for one hidden vector, in float32.

    Keep a 1D input 1D. The previous version indexed x[0] after norm_hidden had
    already returned a 1D tensor when apply_norm=False, turning the input into a
    scalar and making F.linear fail with the 0D/2D error. Whether the result is
    squeezed follows the caller's input, not the shape norm_hidden returns.
    """
    one_d = h.dim() == 1
    x = norm_hidden(h, apply_norm)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"lens_all expected one hidden vector, got shape {tuple(x.shape)}")
    logits = F.linear(x.to(device=W_U.device, dtype=W_U.dtype), W_U).float()
    return logits[0] if one_d else logits


def target_logit(h, tid, apply_norm=True):
    x = norm_hidden(h, apply_norm)
    # norm_hidden returns [1, H] for a 1D input when it applies the norm, but
    # returns the raw 1D vector when it does not. Collapse only the former.
    if x.dim() == 2:
        x = x[0]
    x = x.to(device=W_U.device, dtype=W_U.dtype)
    return float((x.float() * W_U[tid].float()).sum())


def tokenize_pair(text: str, target: str):
    a = tok(text, add_special_tokens=False)["input_ids"]
    b = tok(text + target, add_special_tokens=False)["input_ids"]
    if b[: len(a)] != a or len(b) <= len(a):
        # BPE boundary shifted; fall back to the target's own first token.
        t = tok(target, add_special_tokens=False)["input_ids"]
        b = a + t
    return a, b


def forward(ids, attentions=False):
    t = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode():
        return model(input_ids=t, output_hidden_states=True,
                     output_attentions=attentions, use_cache=False)


# --------------------------------------------------------------------------- #
# stage: capture
# --------------------------------------------------------------------------- #
def capture_prompt(p: dict, full: bool) -> dict:
    global CURRENT, CAPTURE_ON
    text_ids, full_ids = tokenize_pair(p["text"], p["target"])
    T = len(text_ids)
    tid = full_ids[T]
    CURRENT = {}
    CAPTURE_ON = True
    # The final-query attention row is small and is shown for every prompt.
    # Only the DEEP prompts additionally store the full T×T matrices.
    out = forward(full_ids, attentions=True)
    CAPTURE_ON = False

    hs = out.hidden_states
    L = len(hs) - 1
    q = T - 1
    final_lp = F.log_softmax(lens_all(state_at(hs, len(hs) - 1, q), apply_norm=False), dim=-1)

    top, entropy, tprob, trank, top1p, kl, dla_a, dla_m, norms, deltas, cosines = [], [], [], [], [], [], [], [], [], [], []
    for l in range(L + 1):
        h = state_at(hs, l, q)
        logits = lens_all(h, apply_norm=(l < L))
        lp = F.log_softmax(logits, dim=-1)
        probs = lp.exp()
        t_lp = float(lp[tid])
        tprob.append(math.exp(t_lp))
        trank.append(int((logits > logits[tid]).sum()))
        top1p.append(math.exp(float(lp.max())))
        ent = float(-(probs * lp).sum())
        entropy.append(ent)
        kl.append(float((probs * (lp - final_lp)).sum()))
        vals, idxs = torch.topk(probs, 12)
        top.append([[tok.decode([int(i)]), float(v)] for v, i in zip(vals, idxs)])
        norms.append(float(h.float().norm()))
        if l < L:
            h1 = state_at(hs, l + 1, q)
            # A sharded device_map can put adjacent layers on different GPUs.
            h1_local = h1.to(h.device)
            deltas.append(float((h1_local - h).float().norm()))
            cosines.append(float(F.cosine_similarity(h.float(), h1_local.float(), dim=0)))
            a = CURRENT.get("attn", {}).get(l)
            if a is not None:
                ao = a[0, q]
                ao = ao.to(h.device)
                mid = h + ao
                dla_a.append(target_logit(mid, tid) - target_logit(h, tid))
                dla_m.append(target_logit(h1, tid, apply_norm=(l + 1 < L)) - target_logit(mid, tid))
            else:
                dla_a.append(0.0)
                dla_m.append(0.0)

    # probability of the whole continuation, not only its first token
    seq_ids = full_ids[T:T + 8]
    seq_probs, seq_lp = [], 0.0
    for k, t in enumerate(seq_ids):
        lp_t = F.log_softmax(out.logits[0, T - 1 + k].float(), -1)
        seq_probs.append(math.exp(float(lp_t[t])))
        seq_lp += float(lp_t[t])

    entry = dict(
        tokens=[tok.decode([i]) for i in text_ids],
        targetSeqProb=math.exp(seq_lp), targetTokenProbs=r4(seq_probs, 4),
        targetTokens=[tok.decode([int(t)]) for t in seq_ids],
        top=top, entropy=r4(entropy), targetProb=r4(tprob, 5), targetRank=trank,
        top1Prob=r4(top1p, 5), kl=r4(kl),
        finalTop1=tok.decode([int(lens_all(state_at(hs, len(hs) - 1, q), apply_norm=False).argmax())]),
        targetStr=p["target"], n=T, group=p["group"],
        dla=dict(attn=r4(dla_a, 3), mlp=r4(dla_m, 3),
                 resid=r4([target_logit(state_at(hs, l, q), tid, apply_norm=(l < L)) for l in range(L + 1)], 3)),
        norms=dict(norm=r4(norms, 3), delta=r4(deltas, 3), cosine=r4(cosines, 4)),
    )

    # ---- per-position river -------------------------------------------------
    if full:
        rivers = []
        for l in range(L + 1):
            h = seq_at(hs, l)
            logits = F.linear(norm_hidden(h, apply_norm=(l < L)).to(device=W_U.device, dtype=W_U.dtype), W_U).float()
            probs = logits.softmax(-1)
            tprobs = probs[:, tid]
            vals, idxs = probs.max(-1)
            rivers.append(dict(
                top1=[tok.decode([int(i)]) for i in idxs],
                p=r4(vals, 4), pt=r4(tprobs, 4),
            ))
        entry["river"] = rivers
        entry["posNorm"] = r4([seq_at(hs, l).float().norm(dim=-1) for l in range(L + 1)], 3)

    # ---- attention ----------------------------------------------------------
    if out.attentions is not None:
        stats, heads = [], []
        att = out.attentions
        for l, a in enumerate(att):
            # Drop the (causally masked) target positions appended after the
            # prompt. The displayed map is the final prompt query over the prompt.
            row = a[0, :, q, :T].float()  # [H, T prompt positions]
            heads.append(r4(row, 4))
            st = []
            for hh in range(row.shape[0]):
                r = row[hh]
                p_ = r.clamp_min(1e-12)
                ent = float(-(p_ * p_.log()).sum())
                # The attention row can be one position shorter than the input
                # when the model omits the final query (or uses a cache). Derive
                # the length from the row itself rather than assuming T.
                rv = r.detach().cpu().numpy().reshape(-1)
                n = rv.size
                prev = float(r[q - 1]) if 1 <= q < n + 1 else 0.0
                self_ = float(r[q]) if q < n else 0.0
                sink = float(r[0]) if n else 0.0
                # induction: attend to the token after an earlier copy of the
                # current token, averaged over all such earlier copies.
                prev_pos = [i for i in range(min(q, n)) if full_ids[i] == full_ids[q]]
                ind = float(np.mean([float(r[i + 1]) for i in prev_pos if i + 1 < n])) if prev_pos else 0.0
                dist = float(sum(float(r[j]) * (q - j) for j in range(min(T, n))))
                nd = np.arange(n)
                mask = rv > 1e-6
                slope = 0.0
                if mask.sum() > 3:
                    slope = float(np.polyfit(nd[mask], np.log(rv[mask] + 1e-12), 1)[0])
                st.append(dict(prev=prev, self=self_, sink=sink, ent=ent, ind=ind, dist=dist, slope=slope))
            stats.append(st)
        entry["attention"] = dict(
            shape=[len(att), att[0].shape[1], T], heads=heads, stats=stats,
            mean=r4(torch.stack([a[0].mean(0)[q, :T].float().cpu() for a in att]), 4),
        )
        if full:
            entry["attentionFull"] = r4([a[0].float() for a in att], 3)

    # ---- MLA latent ---------------------------------------------------------
    if "kv" in CURRENT:
        kv = [CURRENT["kv"][l][0, q].float().cpu().numpy() for l in sorted(CURRENT["kv"])]
        entry["latent"] = dict(kv=r4(kv, 4), norms=r4([float(np.linalg.norm(v)) for v in kv], 3))
    if "q" in CURRENT:
        qv = [CURRENT["q"][l][0, q].float().cpu().numpy() for l in sorted(CURRENT["q"])]
        entry["qLatent"] = r4([float(np.linalg.norm(v)) for v in qv], 3)

    # ---- router -------------------------------------------------------------
    if "gate" in CURRENT:
        rt, rtop, rent, rmargin, rmax, rload = [], [], [], [], [], {}
        router_dim = 0
        for l in sorted(CURRENT["gate"]):
            scores, idx, wgt = CURRENT["gate"][l]
            E = scores.shape[-1]
            router_dim = int(E)
            load = torch.bincount(idx.reshape(-1).cpu(), minlength=E).tolist()
            rload[l] = load
            rt.append(r4(scores, 4))
            rtop.append([[[int(e), float(w)] for e, w in zip(ii, ww)]
                         for ii, ww in zip(idx.cpu(), wgt.float().cpu())])
            norm = scores / scores.sum(-1, keepdim=True).clamp_min(1e-9)
            rent.append(r4(-(norm * norm.clamp_min(1e-9).log()).sum(-1), 4))
            srt = wgt.float().sort(-1, descending=True).values
            rmargin.append(r4(srt[:, 0] - srt[:, 1] if srt.shape[-1] > 1 else srt[:, 0], 4))
            rmax.append(r4(srt[:, 0], 4))
        entry["router"] = dict(
            shape=[len(rt), T, router_dim], layers=sorted(CURRENT["gate"]), top=rtop,
            entropy=rent, margin=rmargin, max=rmax, load=rload,
        )
        if full:
            entry["routerFull"] = rt

    return entry, tid, T


def stage_capture():
    if stage_done("capture") and (OUT / "meta.json").exists():
        # Do not trust a .done marker from an older, incomplete capture. This
        # is what made the first Kaggle retry reuse a corpus of zero prompts.
        cached_errors = preflight(["capture"])
        if not cached_errors:
            log("stage 'capture' cached")
            return json.loads((OUT / "meta.json").read_text())
        log("cached capture is incomplete; recomputing")
        for err in cached_errors:
            log(f"  - {err}")
    prompts_out, lens_out, attn_out, router_out, norms_out, dla_out = [], {}, {}, {}, {}, {}
    latent_out = {}
    total_tokens = 0
    t0 = time.time()
    wanted = PROMPTS[:LIMIT] if LIMIT else PROMPTS
    for i, p in enumerate(wanted):
        full = p["id"] in DEEP
        try:
            e, tid, T = capture_prompt(p, full)
        except Exception as exc:
            log(f"FAILED {p['id']}: {exc}")
            traceback.print_exc()
            continue
        prompts_out.append(dict(id=p["id"], group=p["group"], text=p["text"], target=p["target"],
                                targetStr=p["target"], targetId=int(tid), tokens=e["tokens"], n=T,
                                finalTop1=e["finalTop1"], top1Prob=e["top1Prob"][-1],
                                targetProb=e["targetProb"][-1], targetRank=e["targetRank"][-1],
                                targetSeqProb=e["targetSeqProb"], targetTokenProbs=e["targetTokenProbs"],
                                targetTokens=e["targetTokens"]))
        if "latent" in e:
            latent_out[p["id"]] = dict(tokens=e["tokens"], kv=e.pop("latent")["kv"],
                                        qNorm=e.pop("qLatent", None))
        lens_out[p["id"]] = {k: e[k] for k in ("tokens", "top", "entropy", "targetProb", "targetRank",
                                                "top1Prob", "kl", "finalTop1", "targetStr", "river",
                                                "targetSeqProb", "targetTokenProbs", "targetTokens") if k in e}
        norms_out[p["id"]] = dict(tokens=e["tokens"], **e["norms"], posNorm=e.get("posNorm"))
        dla_out[p["id"]] = dict(targetStr=p["target"], **e["dla"])
        if "attention" in e:
            a = e["attention"]
            attn_out[p["id"]] = dict(tokens=e["tokens"], shape=a["shape"], heads=a["heads"],
                                     mean=a["mean"], stats=a["stats"])
            if "attentionFull" in e:
                (OUT / f"attentionFull_{p['id']}.json").write_text(json.dumps(
                    dict(tokens=e["tokens"], heads=e["attentionFull"]), separators=(",", ":")))
        if "router" in e:
            router_out[p["id"]] = dict(tokens=e["tokens"], **e["router"])
        total_tokens += T
        log(f"  captured {i+1}/{len(PROMPTS)} {p['id']:14} T={T:2d}  "
            f"P={e['targetProb'][-1]:.3f}  {(time.time()-t0):.0f}s")
        if (i + 1) % 8 == 0:
            gc.collect(); torch.cuda.empty_cache()

    meta = dict(
        modelId=MODEL_ID, modelName=MODEL_ID.split("/")[-1], layers=n_layers(),
        hidden=cfg_int("hidden_size"), heads=cfg_int("num_attention_heads"),
        kvHeads=cfg_int("num_key_value_heads", cfg_int("num_attention_heads")),
        vocab=cfg_int("vocab_size"), experts=cfg_int("n_routed_experts", 0),
        shared=cfg_int("n_shared_experts", 0), topk=cfg_int("num_experts_per_tok", 0),
        moeInter=cfg_int("moe_intermediate_size", 0),
        corpus=len(prompts_out), tokens=total_tokens,
        generatedAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        precision="nf4 (4-bit) weights, fp16 compute" if QUANT == "4bit" else DTYPE,
        quant=QUANT, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        evidence="forward passes only — no gradients, no synthetic values",
        runtimeSec=round(time.time() - t0, 1), deep=DEEP,
    )
    save("meta", meta)
    save("prompts", prompts_out)
    save("lens", lens_out)
    save("attention", attn_out)
    save("router", router_out)
    save("norms", norms_out)
    save("dla", dla_out)
    save("latent_raw", latent_out)
    if len(prompts_out) != len(wanted):
        raise RuntimeError(
            f"capture incomplete: {len(prompts_out)}/{len(wanted)} prompts succeeded; "
            "artefacts were left for inspection but the stage was not cached"
        )
    mark_done("capture")
    return meta


# --------------------------------------------------------------------------- #
# stage: ablation
# --------------------------------------------------------------------------- #
def zero_hook(m, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


def identity_block(m, inp, out):
    if isinstance(out, tuple):
        return (inp[0],) + out[1:]
    return inp[0]


def nll_of(ids, tid):
    t = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(input_ids=t, use_cache=False).logits[0, -1].float()
    return float(-F.log_softmax(logits, -1)[tid])


def metrics_of(ids, tid):
    """Target probability, target logit and target-vs-best-other margin."""
    t = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(input_ids=t, use_cache=False).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    best_other = logits.clone()
    best_other[tid] = -float("inf")
    return dict(prob=float(lp[tid].exp()), logit=float(logits[tid]),
                margin=float(logits[tid] - best_other.max()))


def head_zero_pre_hook(module, args):
    """Zero one head's slice of the attention output before o_proj.

    DeepseekV2Attention reshapes the head outputs to [B, T, heads*v_head_dim]
    and then calls o_proj. A pre-hook on o_proj is therefore a clean head-level
    intervention; it does not touch the other heads or the residual stream.
    """
    x = args[0]
    if not torch.is_tensor(x) or x.shape[-1] % HEAD_ZERO[0] != 0:
        return None
    x = x.clone()
    width = x.shape[-1] // HEAD_ZERO[0]
    h = HEAD_ZERO[1]
    x[..., h * width:(h + 1) * width] = 0
    return (x,) + tuple(args[1:])


HEAD_ZERO = (16, 0)  # (number of query heads, head index); set by the stage


def run_with_zero(specs, ids, tid):
    """Run one forward pass with zero-output hooks on every supplied module."""
    handles = [spec.register_forward_hook(zero_hook) for spec in specs]
    try:
        return metrics_of(ids, tid)
    finally:
        for handle in handles:
            handle.remove()


def stage_ablation():
    got = cached("ablation")
    if got:
        return got
    by_id = {p["id"]: p for p in PROMPTS}
    L = n_layers()
    base, ids_map = {}, {}
    for pid in ABL_PROMPTS:
        p = by_id[pid]
        a, b = tokenize_pair(p["text"], p["target"])
        ids_map[pid] = (b, b[len(a)])
        base[pid] = nll_of(*ids_map[pid])
    log(f"baseline NLL {np.mean(list(base.values())):.3f}")

    layers = []
    t0 = time.time()
    for l in range(L):
        row = dict(l=l)
        for kind in ("attn", "mlp", "block"):
            mod = model.model.layers[l].self_attn if kind == "attn" else (
                model.model.layers[l].mlp if kind == "mlp" else model.model.layers[l])
            hook = mod.register_forward_hook(zero_hook if kind != "block" else identity_block)
            nlls = []
            for pid in ABL_PROMPTS:
                try:
                    nlls.append(nll_of(*ids_map[pid]))
                except Exception:
                    nlls.append(float("nan"))
            hook.remove()
            d = [n - base[p] for n, p in zip(nlls, ABL_PROMPTS)]
            row[kind] = dict(nll=float(np.nanmean(nlls)), delta=float(np.nanmean(d)),
                             perPrompt=r4(d, 3))
        layers.append(row)
        if l % 5 == 0:
            log(f"  ablation L{l:02d}  block ΔNLL {row['block']['delta']:+.3f}  {(time.time()-t0):.0f}s")
            gc.collect(); torch.cuda.empty_cache()

    # expert knockout: zero one routed expert, all tokens routed to it.
    exp_rows = []
    for l in [x for x in ABL_LAYERS if x < L]:
        mlp = model.model.layers[l].mlp
        if not hasattr(mlp, "experts"):
            continue
        for e in ABL_EXPERTS:
            if e >= len(mlp.experts):
                continue
            hook = mlp.experts[e].register_forward_hook(zero_hook)
            ds = []
            for pid in ABL_PROMPTS[:6]:
                try:
                    ds.append(nll_of(*ids_map[pid]) - base[pid])
                except Exception:
                    ds.append(float("nan"))
            hook.remove()
            exp_rows.append(dict(l=l, e=e, delta=float(np.nanmean(ds)), perPrompt=r4(ds, 3)))
        log(f"  expert ablation L{l:02d} done  {(time.time()-t0):.0f}s")

    shared_rows = []
    for l in range(L):
        mlp = model.model.layers[l].mlp
        if not hasattr(mlp, "shared_experts"):
            continue
        hook = mlp.shared_experts.register_forward_hook(zero_hook)
        ds = [nll_of(*ids_map[pid]) - base[pid] for pid in ABL_PROMPTS[:6]]
        hook.remove()
        shared_rows.append(dict(l=l, delta=float(np.mean(ds)), perPrompt=r4(ds, 3)))

    abl = dict(baseline=float(np.mean(list(base.values()))), prompts=ABL_PROMPTS,
               baselineByPrompt=base, layers=layers, experts=exp_rows, shared=shared_rows)
    save("ablation", abl)
    mark_done("ablation")
    return abl


# --------------------------------------------------------------------------- #
# stage: causal head / expert ablation
# --------------------------------------------------------------------------- #
def stage_interventions():
    got = cached("interventions")
    if got:
        return got
    global HEAD_ZERO
    by_id = {p["id"]: p for p in PROMPTS}
    L = n_layers()
    heads = int(getattr(model.config, "num_attention_heads", 16))
    head_dim = int(getattr(model.model.layers[0].self_attn, "v_head_dim", 128))
    ids_map = {}
    base = {}
    for pid in CAUSAL_PROMPTS:
        p = by_id[pid]
        a, b = tokenize_pair(p["text"], p["target"])
        # Score the target at the final prompt position, not after the target
        # has already been appended. Passing b here reads the next-token
        # distribution *after* the answer and silently measures the wrong token.
        ids_map[pid] = (a, b[len(a)])
        base[pid] = metrics_of(*ids_map[pid])
    log("causal baseline: " + ", ".join(
        f"{pid} p={base[pid]['prob']:.3f} m={base[pid]['margin']:+.2f}" for pid in CAUSAL_PROMPTS))

    head_rows = []
    t0 = time.time()
    for l in range(L):
        sa = model.model.layers[l].self_attn
        for h in range(heads):
            HEAD_ZERO = (heads, h)
            handle = sa.o_proj.register_forward_pre_hook(head_zero_pre_hook)
            try:
                vals = [metrics_of(*ids_map[pid]) for pid in CAUSAL_PROMPTS]
            finally:
                handle.remove()
            row = dict(l=l, h=h, headDim=head_dim)
            for key in ("prob", "logit", "margin"):
                row["delta" + key[0].upper() + key[1:]] = float(np.mean([
                    vals[i][key] - base[pid][key] for i, pid in enumerate(CAUSAL_PROMPTS)
                ]))
            row["perPrompt"] = [r4(v["prob"] - base[pid]["prob"], 5)
                                for v, pid in zip(vals, CAUSAL_PROMPTS)]
            head_rows.append(row)
        if l % 5 == 0:
            log(f"  head ablation L{l:02d} ({len(head_rows)} rows) {(time.time()-t0):.0f}s")
            gc.collect(); torch.cuda.empty_cache()

    # Expert ablation across every MoE layer, not only the six in the first sweep.
    expert_rows = []
    exp_ids = CAUSAL_PROMPTS[:2]
    for l in range(L):
        mlp = model.model.layers[l].mlp
        if not hasattr(mlp, "experts"):
            continue
        for e in ABL_EXPERTS:
            if e >= len(mlp.experts):
                continue
            handle = mlp.experts[e].register_forward_hook(zero_hook)
            try:
                vals = [metrics_of(*ids_map[pid]) for pid in exp_ids]
            finally:
                handle.remove()
            expert_rows.append(dict(
                l=l, e=e,
                deltaProb=float(np.mean([vals[i]["prob"] - base[pid]["prob"]
                                         for i, pid in enumerate(exp_ids)])),
                deltaMargin=float(np.mean([vals[i]["margin"] - base[pid]["margin"]
                                           for i, pid in enumerate(exp_ids)])),
                perPrompt=[r4(v["prob"] - base[pid]["prob"], 5)
                           for v, pid in zip(vals, exp_ids)],
            ))
        log(f"  expert ablation L{l:02d} ({len(expert_rows)} rows) {(time.time()-t0):.0f}s")

    def top(rows, key, n=8):
        return [r for r in sorted(rows, key=lambda x: abs(x[key]), reverse=True)[:n]]

    result = dict(
        method="real zero-output ablations on DeepSeek-V2-Lite, forward passes only",
        model=MODEL_ID, quant=QUANT, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        prompts=CAUSAL_PROMPTS, heads=heads, headDim=head_dim, layers=L,
        baseline=base, headRows=head_rows, expertRows=expert_rows,
        topHeads=top(head_rows, "deltaProb", 12), topExperts=top(expert_rows, "deltaProb", 12),
        note=("Attention heads are ablated by zeroing one head's o_proj input; "
              "routed experts by zeroing that expert's output. All token positions "
              "are affected. Deltas are relative to the unmodified model."),
    )
    save("interventions", result)
    mark_done("interventions")
    return result


def edge_specs():
    """Residual-write edges in a 54-edge chain graph.

    a_l -> m_l is the attention write consumed by the MLP. m_l -> a_{l+1} is
    the MLP write consumed by the next attention block. The final MLP has a
    direct edge to the readout. Token-to-attention edges are intentionally not
    included: this is a clean component-output graph, not a token-routing graph.
    """
    L = n_layers()
    specs = []
    for l in range(L):
        specs.append(dict(id=f"a{l}->m{l}", s=f"a{l}", t=f"m{l}", l=l,
                          kind="attn", module=model.model.layers[l].self_attn))
        if l + 1 < L:
            specs.append(dict(id=f"m{l}->a{l+1}", s=f"m{l}", t=f"a{l+1}", l=l,
                              kind="mlp", module=model.model.layers[l].mlp))
    specs.append(dict(id=f"m{L-1}->readout", s=f"m{L-1}", t="readout", l=L - 1,
                      kind="mlp", module=model.model.layers[L - 1].mlp))
    return specs


def stage_causal_edges():
    got = cached("edges")
    if got:
        return got
    by_id = {p["id"]: p for p in PROMPTS}
    specs = edge_specs()
    id_to_spec = {s["id"]: s for s in specs}
    pid = EDGE_DISCOVERY
    p = by_id[pid]
    a, b = tokenize_pair(p["text"], p["target"])
    # The edge metric must read the target from the final prompt position.
    ids, tid = a, b[len(a)]
    base = metrics_of(ids, tid)
    log(f"edge baseline {pid}: p={base['prob']:.3f} margin={base['margin']:+.2f} "
        f"({len(specs)} candidates)")

    kept = list(specs)
    removed = []
    steps = []
    t0 = time.time()
    while len(kept) > EDGE_BUDGET:
        best = None
        for spec in kept:
            try:
                m = run_with_zero([spec["module"]], ids, tid)
            except Exception as exc:
                log(f"  edge {spec['id']} failed: {exc}")
                continue
            score = abs(m["prob"] - base["prob"])
            if best is None or score < best[0]:
                best = (score, spec, m)
        if best is None:
            raise RuntimeError("every candidate edge failed")
        _, spec, m = best
        kept.remove(spec)
        removed.append(spec)
        steps.append(dict(step=len(steps) + 1, edge=spec["id"], kind=spec["kind"],
                          layer=spec["l"], deltaProb=float(m["prob"] - base["prob"]),
                          prob=m["prob"], kept=len(kept)))
        if len(steps) % 6 == 0:
            log(f"  pruned {len(steps):02d}, kept {len(kept):02d}, "
                f"last {spec['id']} Δp={m['prob']-base['prob']:+.4f} {(time.time()-t0):.0f}s")

    # Validate the discovered circuit on the held-out prompt by ablating every
    # removed edge simultaneously. This is a real forward pass, not a replay.
    validation = None
    if EDGE_VALIDATION in by_id:
        vp = by_id[EDGE_VALIDATION]
        va, vb = tokenize_pair(vp["text"], vp["target"])
        vids, vtid = va, vb[len(va)]
        vbase = metrics_of(vids, vtid)
        vcircuit = run_with_zero([s["module"] for s in removed], vids, vtid)
        validation = dict(prompt=EDGE_VALIDATION, baseline=vbase, circuit=vcircuit,
                          retention=vcircuit["prob"] / max(vbase["prob"], 1e-12))

    discovery_circuit = run_with_zero([s["module"] for s in removed], ids, tid)
    result = dict(
        method="real greedy edge ablation on V2-Lite: zero-output residual-write edges, forward passes only",
        model=MODEL_ID, quant=QUANT, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        prompts=dict(discovery=pid, validation=EDGE_VALIDATION),
        candidateCount=len(specs), budget=EDGE_BUDGET,
        baseline=base, discoveryCircuit=discovery_circuit,
        discoveryRetention=discovery_circuit["prob"] / max(base["prob"], 1e-12),
        validation=validation, steps=steps,
        keptEdges=[dict(id=s["id"], s=s["s"], t=s["t"], l=s["l"], kind=s["kind"]) for s in kept],
        removedEdges=[dict(id=s["id"], s=s["s"], t=s["t"], l=s["l"], kind=s["kind"]) for s in removed],
        note=("Greedy ACDC-style search with a zero-ablation reference. "
              "An edge is the write from one component to the next residual consumer; "
              "token-to-attention edges and path patching are out of scope. "
              "The held-out number ablates all removed edges at once."),
    )
    save("edges", result)
    mark_done("edges")
    return result


# --------------------------------------------------------------------------- #
# stage: probes + geometry
# --------------------------------------------------------------------------- #
def stage_probes():
    got = cached("probes")
    if got:
        return got
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    L = n_layers()
    # states[prop][layer] = list of vectors, labels[prop] = list, groups[prop]
    states = {p["key"]: [[] for _ in range(L + 1)] for p in PROBE_PROPS}
    labels = {p["key"]: [] for p in PROBE_PROPS}
    groups = {p["key"]: [] for p in PROBE_PROPS}

    for i, p in enumerate(PROBES[: LIMIT * 4] if LIMIT else PROBES):
        ids = tok(p["text"], add_special_tokens=False)["input_ids"]
        out = forward(ids, attentions=False)
        for l in range(L + 1):
            h = out.hidden_states[l][0, -4:].float().cpu().numpy()
            states[p["prop"]][l].extend(h)
        for _ in range(min(4, len(ids))):
            labels[p["prop"]].append(p["label"])
            groups[p["prop"]].append(p["group"])
        if (i + 1) % 24 == 0:
            log(f"  probe forward {i+1}/{len(PROBES)}")

    layers = list(range(L + 1))
    by_prop, std_prop = {}, {}
    for prop in PROBE_PROPS:
        key = prop["key"]
        Xall = states[key]
        y = np.array(labels[key])
        g = np.array(groups[key])
        accs, stds = [], []
        for l in layers:
            X = np.array(Xall[l], dtype=np.float32)
            n_splits = min(5, len(np.unique(g)))
            gkf = GroupKFold(n_splits=n_splits)
            scores = []
            for tr, te in gkf.split(X, y, g):
                pipe = make_pipeline(StandardScaler(),
                                     PCA(n_components=min(32, len(tr) - 1)),
                                     LogisticRegression(max_iter=1000, C=0.5))
                pipe.fit(X[tr], y[tr])
                scores.append(pipe.score(X[te], y[te]))
            accs.append(float(np.mean(scores)))
            stds.append(float(np.std(scores)))
        by_prop[key] = r4(accs, 4)
        std_prop[key] = r4(stds, 4)
        log(f"  probe {key:10} layer0 {accs[0]:.2f} → layerL {accs[-1]:.2f}  chance {prop['chance']}")

    probes = dict(properties=PROBE_PROPS, layers=layers, byProp=by_prop, std=std_prop,
                  nPrompts=len(PROBES), positions=4,
                  method="StandardScaler → PCA(32, fit within fold) → LogisticRegression(C=0.5), GroupKFold(5) by prompt")
    save("probes", probes)
    mark_done("probes")
    return probes


def stage_geometry():
    got = cached("geometry")
    if got:
        return got
    from numpy.linalg import eigh

    L = n_layers()
    ids = [p["id"] for p in PROMPTS]
    H = np.zeros((len(ids), L + 1, cfg_int("hidden_size")), dtype=np.float32)
    traj = {}
    for i, p in enumerate(PROMPTS):
        a, _ = tokenize_pair(p["text"], p["target"])
        out = forward(a, attentions=False)
        for l in range(L + 1):
            H[i, l] = out.hidden_states[l][0, -1].float().cpu().numpy()
        if p["id"] in DEEP:
            traj[p["id"]] = dict(
                tokens=[tok.decode([t]) for t in a],
                points=np.stack([out.hidden_states[l][0].float().cpu().numpy()
                                 for l in range(L + 1)]),
            )

    @torch.no_grad()
    def to_t(x):
        return torch.tensor(x)

    Xs = [to_t(H[:, l, :]) for l in range(L + 1)]
    cka = np.zeros((L + 1, L + 1))
    for i in range(L + 1):
        for j in range(i, L + 1):
            A, B = Xs[i] - Xs[i].mean(0), Xs[j] - Xs[j].mean(0)
            A, B = A / A.norm().clamp_min(1e-12), B / B.norm().clamp_min(1e-12)
            num = (A @ B.T).pow(2).sum()
            den = (A @ A.T).pow(2).sum().sqrt() * (B @ B.T).pow(2).sum().sqrt()
            cka[i, j] = cka[j, i] = float(num / den.clamp_min(1e-12))

    eff, drift = [], []
    for l in range(L + 1):
        X = Xs[l] - Xs[l].mean(0)
        ev = torch.linalg.eigvalsh(X @ X.T).clamp_min(0).numpy()
        ev = np.sort(ev)[::-1]
        eff.append(float((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12)))
        if l:
            drift.append(float(F.cosine_similarity(Xs[l][-1], Xs[l - 1][-1], dim=0)))
    # shared 2-D basis over every layer and prompt (fit once, project everywhere)
    allX = np.concatenate([H[:, l, :] for l in range(L + 1)], 0)
    mu = allX.mean(0)
    U, S, Vt = np.linalg.svd(allX - mu, full_matrices=False)
    basis = Vt[:2].T
    pca = np.stack([((H[:, l, :] - mu) @ basis) for l in range(L + 1)])
    trajectories = {}
    for pid, d in traj.items():
        P = d["points"]  # [L+1, T, d]
        trajectories[pid] = dict(tokens=d["tokens"],
                                 points=np.stack([(P[l] - mu) @ basis for l in range(L + 1)]))
    geo = dict(
        cka=r4(cka, 4), effectiveRank=r4(eff, 3), drift=r4(drift, 4),
        promptIds=ids, pca=dict(points=r4(pca, 3), explained=r4((S[:2] ** 2 / (S ** 2).sum()), 4)),
        tokenTrajectories={k: dict(tokens=v["tokens"], points=r4(v["points"], 3)) for k, v in trajectories.items()},
        note="CKA and effective rank are computed on the 27 last-token states; with n=27 they are descriptive, not asymptotic.",
    )
    save("geometry", geo)
    mark_done("geometry")
    return geo


# --------------------------------------------------------------------------- #
# stage: latent geometry (the compressed KV vector itself)
# --------------------------------------------------------------------------- #
def stage_latent():
    got = cached("latent")
    if got:
        return got
    raw = json.loads((OUT / "latent_raw.json").read_text())
    ids = [p["id"] for p in PROMPTS if p["id"] in raw]
    if not ids:
        return {}
    L = len(raw[ids[0]]["kv"])
    D = cfg_int("kv_lora_rank") or cfg_int("head_dim")
    mats, norms, rope = [], [], []
    for l in range(L):
        X = np.array([raw[i]["kv"][l][:D] for i in ids], dtype=np.float32)
        mats.append(X)
        norms.append(np.linalg.norm(X, axis=1))
        rope.append(np.linalg.norm(np.array([raw[i]["kv"][l][D:] for i in ids], dtype=np.float32), axis=1))

    eff, var_share, dim_var = [], [], []
    grams = []
    for X in mats:
        Xc = X - X.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)
        e = (s ** 2).sum() ** 2 / max(((s ** 2) ** 2).sum(), 1e-12)
        eff.append(float(e))
        var_share.append(float(s[0] ** 2 / max((s ** 2).sum(), 1e-12)))
        dv = X.var(0)
        dim_var.append(dv / max(dv.sum(), 1e-12))
        grams.append(Xc / max(np.linalg.norm(Xc), 1e-12))

    cka = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            num = float((grams[i] @ grams[i].T * (grams[j] @ grams[j].T)).sum())
            cka[i, j] = cka[j, i] = num

    stacked = np.concatenate([(X - X.mean(0)) / max(np.linalg.norm(X - X.mean(0)), 1e-12) for X in mats], 0)
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    basis = Vt[:2].T
    points = np.stack([((X - X.mean(0)) @ basis) for X in mats])

    out = dict(ids=ids, layers=L, dim=D, effectiveRank=r4(eff, 3), varShare=r4(var_share, 4),
               norms=r4(norms, 3), ropeNorm=r4(rope, 3), cka=r4(cka, 4),
               pca=dict(points=r4(points, 3), explained=r4(S[:2] ** 2 / (S ** 2).sum(), 4)),
               dimVar=r4(dim_var, 5),
               note="The KV latent is captured at the last query position of every prompt; geometry is across the corpus, per layer.")
    save("latent", out)
    mark_done("latent")
    return out


# --------------------------------------------------------------------------- #
# stage: context
# --------------------------------------------------------------------------- #
def build_needle(n_tokens: int, depth: float):
    # rough 1.3 tokens/word; build then truncate to the requested budget
    words = FILLER.split()
    body = " ".join(words * (1 + n_tokens // len(words) + 2))
    ids_filler = tok(body, add_special_tokens=False)["input_ids"]
    ids_needle = tok(" " + NEEDLE, add_special_tokens=False)["input_ids"]
    ids_q = tok(NEEDLE_Q, add_special_tokens=False)["input_ids"]
    ids_target = tok(NEEDLE_TARGET, add_special_tokens=False)["input_ids"]
    room = max(32, n_tokens - len(ids_needle) - len(ids_q))
    filler = ids_filler[:room]
    cut = min(max(4, int(depth * (len(filler) - 1))), len(filler) - 1)
    ids = filler[:cut] + ids_needle + filler[cut:] + ids_q
    return ids, len(filler[:cut]) + len(ids_needle), ids_target[0]


def stage_context():
    got = cached("context")
    if got:
        return got
    results = []
    att_saved = None
    for length in NEEDLE_LENGTHS:
        for depth in NEEDLE_DEPTHS:
            ids, needle_end, tid = build_needle(length, depth)
            try:
                out = forward(ids, attentions=(length == 512 and depth == 0.5))
            except RuntimeError as exc:
                log(f"  context {length} d{depth} OOM: {exc}")
                gc.collect(); torch.cuda.empty_cache()
                continue
            lp = F.log_softmax(out.logits[0, -1].float(), -1)
            rank = int((out.logits[0, -1] > out.logits[0, -1, tid]).sum())
            results.append(dict(len=length, depth=depth, n=len(ids),
                                targetProb=float(lp[tid].exp()), targetRank=rank,
                                top1=tok.decode([int(out.logits[0, -1].argmax())]),
                                needlePos=needle_end / len(ids)))
            if out.attentions is not None:
                L = n_layers()
                last = torch.stack([a[0, :, -1, :].float().mean(0).cpu() for a in out.attentions]).numpy()
                att_saved = dict(len=len(ids), meanByLayer=r4(last, 5),
                                 peak=r4(float(lp[tid].exp()), 4))
            del out
            gc.collect(); torch.cuda.empty_cache()
            log(f"  needle len={length:5d} depth={depth:.1f} P={results[-1]['targetProb']:.3f} "
                f"rank={rank}")
    ctx = dict(results=results, attention=att_saved,
               method="answer probability at the final token; one context length captured with attention to show distance decay")
    save("context", ctx)
    mark_done("context")
    return ctx


# --------------------------------------------------------------------------- #
# stage: graph
# --------------------------------------------------------------------------- #
def stage_graph():
    got = cached("graph")
    if got:
        return got
    lens = json.loads((OUT / "lens.json").read_text())
    att = json.loads((OUT / "attention.json").read_text())
    dla = json.loads((OUT / "dla.json").read_text())
    by_id = {p["id"]: p for p in PROMPTS}
    graph = {}
    for pid in DEEP:
        p = by_id[pid]
        a = att.get(pid)
        if not a:
            continue
        d = dla[pid]
        L = len(d["attn"])
        # The attention rows are trimmed to the prompt positions, so the key
        # index and the prompt-token index are aligned.
        key_labels = a["tokens"]
        nodes, edges = [], []
        for l in range(L):
            nodes.append(dict(id=f"a{l}", layer=l, kind="attn", attr=d["attn"][l]))
            nodes.append(dict(id=f"m{l}", layer=l, kind="mlp", attr=d["mlp"][l]))
            edges.append(dict(s=f"a{l}", t=f"m{l}", w=1, kind="chain"))
            if l:
                edges.append(dict(s=f"m{l-1}", t=f"a{l}", w=1, kind="chain"))
            row = a["mean"][l]
            top = sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:3]
            for i in top:
                if row[i] > 0.02 and i < len(key_labels):
                    nodes.append(dict(id=f"t{l}-{i}", layer=l, kind="token", attr=row[i],
                                      token=key_labels[i], pos=i))
                    edges.append(dict(s=f"t{l}-{i}", t=f"a{l}", w=row[i], kind="attn"))
        graph[pid] = dict(id=pid, tokens=a["tokens"], target=p["target"], nodes=nodes, edges=edges,
                          attr=d, shape=[L, n_layers()])
    save("graph", graph)
    mark_done("graph")
    return graph


# --------------------------------------------------------------------------- #
# experts: routing-derived specialisation (no weight surgery)
# --------------------------------------------------------------------------- #
def stage_experts():
    got = cached("experts")
    if got:
        return got
    router = json.loads((OUT / "router.json").read_text())
    prompts = json.loads((OUT / "prompts.json").read_text())
    group_of = {p["id"]: p["group"] for p in prompts}
    groups = sorted(set(group_of.values()))
    L = n_layers()
    E = cfg_int("n_routed_experts", 0)
    if not E:
        return {}
    load = np.zeros((L, E), dtype=np.int64)
    group_load = {g: np.zeros((L, E), dtype=np.int64) for g in groups}
    group_tokens = {g: 0 for g in groups}
    weights = np.zeros((L, E), dtype=np.float64)
    weight_n = np.zeros((L, E), dtype=np.float64)
    profiles = []

    for pid, r in router.items():
        g = group_of.get(pid, "other")
        tops = r["top"]
        # The first_k_dense_replace layers have no router. Older captures did
        # not record layer ids; infer the tail block, which is where MoE starts.
        gate_layers = r.get("layers") or list(range(L - len(tops), L))
        for l, top in zip(gate_layers, tops):
            group_tokens[g] += len(top)
            for pos in top:
                for e, w in pos:
                    load[l, e] += 1
                    group_load[g][l, e] += 1
                    weights[l, e] += w
                    weight_n[l, e] += 1

    for l in range(L):
        for e in range(E):
            tot = load[l, e]
            gs = {g: int(group_load[g][l, e]) for g in groups}
            shares = {g: (gs[g] / group_tokens[g]) for g in groups if group_tokens[g]}
            top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
            profiles.append(dict(l=l, e=e, count=int(tot),
                                 share=float(tot / max(load[l].sum(), 1)),
                                 meanWeight=float(weights[l, e] / max(weight_n[l, e], 1)),
                                 groups={g: r4(v, 5) for g, v in gs.items()},
                                 topGroups=[[g, r4(v, 5)] for g, v in top]))

    co = np.zeros((E, E), dtype=np.int64)
    for pid, r in router.items():
        for top in r["top"]:
            for pos in top:
                ids = [e for e, w in pos]
                for i in ids:
                    for j in ids:
                        if i != j:
                            co[i, j] += 1
    co = co / np.maximum(co.sum(axis=1, keepdims=True), 1)
    save("experts", dict(layers=L, experts=E, load=load.tolist(),
                         groupLoad={g: group_load[g].tolist() for g in groups},
                         profiles=profiles, coactivation=r4(co, 4), groups=groups))
    mark_done("experts")
    return {}


# --------------------------------------------------------------------------- #
def main():
    stages = STAGES.split(",") if STAGES != "all" else ["capture", "ablation", "probes", "geometry", "latent", "context", "graph", "experts", "interventions", "edges"]
    if "all" in stages:
        stages = ["capture", "ablation", "probes", "geometry", "latent", "context", "graph", "experts", "interventions", "edges"]
    load_model()
    register_hooks()

    if "capture" in stages:
        stage_capture()
    if "ablation" in stages:
        try:
            stage_ablation()
        except Exception as exc:
            log(f"ablation failed: {exc}"); traceback.print_exc()
    if "probes" in stages:
        try:
            stage_probes()
        except Exception as exc:
            log(f"probes failed: {exc}"); traceback.print_exc()
    if "geometry" in stages:
        try:
            stage_geometry()
        except Exception as exc:
            log(f"geometry failed: {exc}"); traceback.print_exc()
    if "context" in stages:
        try:
            stage_context()
        except Exception as exc:
            log(f"context failed: {exc}"); traceback.print_exc()
    if "latent" in stages:
        try:
            stage_latent()
        except Exception as exc:
            log(f"latent failed: {exc}"); traceback.print_exc()
    if "graph" in stages:
        try:
            stage_graph()
        except Exception as exc:
            log(f"graph failed: {exc}"); traceback.print_exc()
    if "experts" in stages:
        try:
            stage_experts()
        except Exception as exc:
            log(f"experts failed: {exc}"); traceback.print_exc()
    if "interventions" in stages:
        try:
            stage_interventions()
        except Exception as exc:
            log(f"interventions failed: {exc}"); traceback.print_exc()
    if "edges" in stages:
        try:
            stage_causal_edges()
        except Exception as exc:
            log(f"edges failed: {exc}"); traceback.print_exc()
    if preflight(stages):
        raise SystemExit(1)
    log("done")


if __name__ == "__main__":
    main()
