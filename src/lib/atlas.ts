/// <reference types="vite/client" />
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------------------------------------------ types */
export type FamilyMember = {
  id: string;
  name: string;
  short: string;
  generation: string;
  date: string;
  published_total: number;
  published_active: number;
  computed_total: number;
  computed_active: number;
  total_error: number;
  active_error: number;
  context: number;
  trained: string;
  license: string;
  attention: string;
  precision: string;
  mtp: number;
  hash_layers: number;
  note: string;
  source: string;
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  vocab: number;
  experts: number;
  shared: number;
  topk: number;
  moeInter: number;
  headDim: number;
  qLora: number | null;
  kvLora: number | null;
  indexTopk: number | null;
  sliding: number | null;
  compress: number[] | null;
  kvSource: number[] | null;
  sparsity: number;
  kv: {
    scheme: string; per_token: number; mha: number; gqa: number;
    ratio: number; layers_stored: number; sliding: number; per_layer: number[];
  };
  counts: Record<string, number>;
  plan: LayerPlan[];
};

export type LayerPlan = {
  layer: number;
  ffn: "dense" | "moe";
  attn: "mla" | "dsa" | "csa" | "hca" | "sliding";
  compress: number;
  hash: boolean;
  kv_shared: boolean;
  kv_source: boolean;
};

export type Family = {
  generatedAt: string;
  thesis: string;
  members: FamilyMember[];
  generations: { id: string; label: string; change: string; experts: number; active: number }[];
  sources: { name: string; url: string }[];
};

export type ArchFocus = {
  id: string; name: string; layers: number; hidden: number; heads: number;
  kvHeads: number; headDim: number; qLora: number | null; kvLora: number | null;
  vocab: number; experts: number; shared: number; topk: number; moeInter: number;
  denseInter: number; firstDense: number; mtp: number; plan: LayerPlan[];
  attention: { q: number; kv_a: number; kv_b: number; o: number; norm: number; total: number };
  tie: boolean; scoring: string; topkMethod: string; routedScale: number;
  normTopk: boolean; indexTopk: number | null; sliding: number | null;
  hashLayers: number; nGroups: number | null; topkGroup: number | null;
  qkRope: number; qkNope: number; vHead: number;
  oGroups: number | null; oLora: number | null;
};

export type Architecture = {
  focus: Record<string, ArchFocus>;
  comparison: { id: string; name: string; layers: number; hidden: number; experts: number; topk: number; active: number; total: number; context: number }[];
  assumptions: string[];
};

export type BudgetMember = {
  id: string; name: string; short: string; generation: string;
  total: number; active: number; published_total: number; published_active: number;
  sparsity: number; layers: number; experts: number; topk: number; shared: number;
  attn: number; moe: number; dense: number; embed: number; lmHead: number; norms: number;
  nMoe: number; nDense: number; sharedTotal: number; routedTotal: number;
  attnPerLayer: number; routedPerLayer: number; sharedPerLayer: number; gatePerLayer: number;
  perExpert: number; expertShare: number; activeShare: number; flopsPerToken: number;
  attnFlops: Record<string, number>; denseEquivalent: number;
};

export type Budget = { members: BudgetMember[] };

export type CacheMember = {
  id: string; name: string; short: string; generation: string; scheme: string;
  perToken: number; officialPerToken: number | null; mha: number; gqa: number; ratio: number;
  layersStored: number; sliding: number; layers: number; kvHeads: number;
  headDim: number; kvLora: number | null; compress: number[] | null;
  kvSource: number[] | null; perLayer: number[];
  at: Record<string, number>;
};

export type Cache = {
  members: CacheMember[]; contexts: number[];
  official: { model: string; claim: string; url: string }[];
};

export type FrontierMember = {
  id: string; name: string; short: string; generation: string; attention: string;
  context: number; layers: number; heads: number; qk: number; v: number;
  kvPerToken: number; kv1m: number;
  indexTopk: number | null; sliding: number | null;
  compress: number[] | null; kvSource: number[] | null; plan: LayerPlan[];
  flops: Record<string, number>;
};

export type Frontier = { members: FrontierMember[]; contexts: number[] };

export type Meta = {
  modelId: string; modelName: string; layers: number; hidden: number; heads: number;
  kvHeads: number; vocab: number; experts: number; shared: number; topk: number;
  moeInter: number; corpus: number; tokens: number; generatedAt: string;
  precision: string; quant: string; device: string; evidence: string;
  runtimeSec: number; deep: string[];
};

export type Prompt = {
  id: string; group: string; text: string; target: string; targetStr: string;
  targetId: number; tokens: string[]; n: number; finalTop1: string;
  top1Prob: number; targetProb: number; targetRank: number;
  targetSeqProb: number; targetTokenProbs: number[]; targetTokens: string[];
};

export type LensEntry = {
  tokens: string[];
  top: [string, number][][];
  entropy: number[]; targetProb: number[]; targetRank: number[];
  top1Prob: number[]; kl: number[]; finalTop1: string; targetStr: string;
  targetSeqProb?: number; targetTokenProbs?: number[]; targetTokens?: string[];
  river?: { top1: string[]; p: number[]; pt: number[] }[];
};

export type HeadStat = {
  prev: number; self: number; sink: number; ent: number; ind: number; dist: number; slope: number;
};

export type AttentionEntry = {
  tokens: string[];
  /** [layers, heads, keys] at the final query position. */
  shape: [number, number, number];
  heads: number[][][];
  mean: number[][];
  stats: HeadStat[][];
};

export type RouterEntry = {
  tokens: string[];
  shape: [number, number, number];
  layers?: number[];
  top: [number, number][][][];
  entropy: number[][]; margin: number[][]; max: number[][];
  load: Record<string, number[]>;
};

export type NormsEntry = {
  tokens: string[];
  norm: number[]; delta: number[]; cosine: number[];
  attn?: number[]; mlp?: number[];
  posNorm?: number[][];
};

export type DlaEntry = {
  targetStr: string;
  attn: number[]; mlp: number[]; resid: number[];
};

export type AblationLayer = {
  l: number;
  attn: { nll: number; delta: number; perPrompt: number[] };
  mlp: { nll: number; delta: number; perPrompt: number[] };
  block: { nll: number; delta: number; perPrompt: number[] };
};

export type Ablation = {
  baseline: number; prompts: string[]; baselineByPrompt: Record<string, number>;
  layers: AblationLayer[];
  experts: { l: number; e: number; delta: number; perPrompt: number[] }[];
  shared: { l: number; delta: number; perPrompt: number[] }[];
};

export type Probes = {
  properties: { key: string; label: string; chance: number; classes: string[] }[];
  layers: number[];
  byProp: Record<string, number[]>;
  std: Record<string, number[]>;
  nPrompts: number; positions: number; method: string;
};

export type Geometry = {
  cka: number[][];
  effectiveRank: number[];
  drift: number[];
  promptIds: string[];
  pca: { points: number[][][]; explained: number[] };
  tokenTrajectories: Record<string, { tokens: string[]; points: number[][][] }>;
  note: string;
};

export type Latent = {
  ids: string[]; layers: number; dim: number;
  effectiveRank: number[]; varShare: number[]; norms: number[][];
  ropeNorm: number[][]; cka: number[][];
  pca: { points: number[][][]; explained: number[] };
  dimVar: number[][];
  note: string;
};

export type Experts = {
  layers: number; experts: number;
  load: number[][];
  groupLoad: Record<string, number[][]>;
  profiles: {
    l: number; e: number; count: number; share: number; meanWeight: number;
    groups: Record<string, number>; topGroups: [string, number][];
  }[];
  coactivation: number[][];
  groups: string[];
};

export type GraphNode = {
  id: string; layer: number; kind: "attn" | "mlp" | "token"; attr: number;
  token?: string; pos?: number;
};
export type GraphEdge = { s: string; t: string; w: number; kind: string };
export type MechanismAudit = {
  generatedAt: string;
  method: string;
  nPrompts: number;
  nAblationPrompts: number;
  layers: number;
  agreement: {
    attnSpearman: number;
    mlpSpearman: number;
    blockSpearman: number;
    attnTop5Overlap: number;
    mlpTop5Overlap: number;
    blockTop5Overlap: number;
  };
  top: Record<string, { layer: number; value: number }[]>;
  table: {
    layer: number;
    attnDla: number;
    mlpDla: number;
    attnAblation: number;
    mlpAblation: number;
    blockAblation: number;
  }[];
  finding: string;
};

export type AcdcAudit = {
  generatedAt: string;
  method: string;
  causal: boolean;
  budget: number;
  candidateEdges: number[];
  prompts: {
    id: string;
    baseline: number;
    final: number;
    retention: number;
    candidates: number;
    kept: number;
    chainKept: number;
    attnKept: number;
    removed: number;
  }[];
  curves: { id: string; [budget: string]: number | string }[];
  aggregate: {
    meanRetention: number;
    minRetention: number;
    maxRetention: number;
    stableEdges: number;
  };
  caveat: string;
};

export type CausalMetric = { prob: number; logit: number; margin: number };

export type CausalHeadRow = {
  l: number; h: number; headDim: number;
  deltaProb: number; deltaLogit: number; deltaMargin: number;
  perPrompt: number[];
};

export type CausalExpertRow = {
  l: number; e: number; deltaProb: number; deltaMargin: number;
  perPrompt: number[];
};

export type CausalInterventions = {
  method: string;
  model: string;
  quant: string;
  device: string;
  prompts: string[];
  heads: number;
  headDim: number;
  layers: number;
  baseline: Record<string, CausalMetric>;
  headRows: CausalHeadRow[];
  expertRows: CausalExpertRow[];
  topHeads: CausalHeadRow[];
  topExperts: CausalExpertRow[];
  note: string;
};

export type CausalEdge = { id: string; s: string; t: string; l: number; kind: string };

export type EdgeCircuit = {
  method: string;
  model: string;
  quant: string;
  device: string;
  prompts: { discovery: string; validation: string };
  candidateCount: number;
  budget: number;
  baseline: CausalMetric;
  discoveryCircuit: CausalMetric;
  discoveryRetention: number;
  validation: { prompt: string; baseline: CausalMetric; circuit: CausalMetric; retention: number } | null;
  steps: { step: number; edge: string; kind: string; layer: number; deltaProb: number; prob: number; kept: number }[];
  keptEdges: CausalEdge[];
  removedEdges: CausalEdge[];
  note: string;
};

export type GraphEntry = {
  id: string; tokens: string[]; target: string;
  nodes: GraphNode[]; edges: GraphEdge[];
  attr: DlaEntry; shape: [number, number];
};

export type ContextData = {
  results: {
    len: number; depth: number; n: number; targetProb: number;
    targetRank: number; top1: string; needlePos: number;
  }[];
  attention: { len: number; meanByLayer: number[][]; peak: number } | null;
  method: string;
};

/* ---------------------------------------------------------------- loading */
const cache = new Map<string, unknown>();
const inflight = new Map<string, Promise<unknown>>();

export async function loadJson<T>(name: string): Promise<T> {
  if (cache.has(name)) return cache.get(name) as T;
  if (inflight.has(name)) return inflight.get(name) as Promise<T>;
  const p = (async () => {
    const res = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`);
    if (!res.ok) throw new Error(`${name}.json · HTTP ${res.status}`);
    const j = await res.json();
    cache.set(name, j);
    return j;
  })();
  inflight.set(name, p);
  // A rejected load must not poison the cache: remove it on either settle.
  // Successful values are already in `cache`, so this only affects retries.
  p.then(() => inflight.delete(name), () => inflight.delete(name));
  return p as Promise<T>;
}

export function useJson<T>(name: string | string[]) {
  const names = Array.isArray(name) ? name : [name];
  const key = names.join("|");
  const [state, setState] = useState<{
    data: Record<string, T> | null;
    error: string | null;
    loading: boolean;
  }>({ data: null, error: null, loading: true });

  useEffect(() => {
    let live = true;
    setState({ data: null, error: null, loading: true });
    Promise.all(names.map((n) => loadJson<T>(n)))
      .then((vals) => {
        if (!live) return;
        const data: Record<string, T> = {};
        names.forEach((n, i) => (data[n] = vals[i]));
        setState({ data, error: null, loading: false });
      })
      .catch((e) => live && setState({ data: null, error: String(e.message ?? e), loading: false }));
    return () => {
      live = false;
    };
  }, [key]);

  return state;
}

export function useOne<T>(name: string) {
  const s = useJson<T>(name);
  return { data: s.data ? (s.data[name] as T) : null, loading: s.loading, error: s.error };
}

/* ------------------------------------------------------------ availability */
export const MEASURED_FILES = [
  "meta", "prompts", "lens", "attention", "router", "experts", "norms",
  "dla", "ablation", "probes", "geometry", "graph", "context", "latent",
  "mechanism", "acdc", "interventions", "edges",
] as const;

export const CHAPTER_FILES: Record<string, string[]> = {
  lens: ["meta", "prompts", "lens"],
  understanding: ["lens", "attention", "router", "dla", "context"],
  commitment: ["prompts", "lens"],
  anatomy: ["norms", "dla"],
  routing: ["router"],
  experts: ["experts"],
  attention: ["attention"],
  latent: ["latent"],
  ablation: ["ablation"],
  circuit: ["graph"],
  geometry: ["geometry"],
  probes: ["probes"],
  context: ["context"],
  mechanism: ["mechanism"],
  acdc: ["acdc"],
  "causal-ablation": ["interventions"],
  "edge-circuit": ["edges"],
};

export type Availability = {
  known: boolean;
  present: string[];
  missing: string[];
  measured: boolean;
  complete: boolean;
};

export function useAvailability(): Availability {
  const [present, setPresent] = useState<string[] | null>(null);
  useEffect(() => {
    let live = true;
    Promise.all(MEASURED_FILES.map(async (n) => {
      try {
        const r = await fetch(`${import.meta.env.BASE_URL}data/${n}.json`, {
          method: "HEAD", cache: "no-store",
        });
        const type = r.headers.get("content-type") ?? "";
        return r.ok && type.includes("json") ? n : null;
      } catch {
        return null;
      }
    })).then((rows) => {
      if (!live) return;
      const found: string[] = [];
      for (const r of rows) if (r) found.push(r);
      setPresent(found);
    });
    return () => { live = false; };
  }, []);
  const have = present ?? [];
  const missing = MEASURED_FILES.filter((n) => !have.includes(n));
  return {
    known: present !== null,
    present: have,
    missing,
    measured: have.length > 0,
    complete: have.length === MEASURED_FILES.length,
  };
}

/* ---------------------------------------------------------------- routing */
export type Chapter = {
  slug: string; n: string; part: string; title: string; short: string;
  desc: string; datum: "config" | "measured" | "mixed" | "method";
};

export const CHAPTERS: Chapter[] = [
  { slug: "", n: "00", part: "Front", title: "Overview", short: "Overview", desc: "Twenty-two instruments, one model family, and the rule that governs every figure.", datum: "mixed" },
  { slug: "family", n: "01", part: "I", title: "Nine machines, one lineage", short: "The family", desc: "Every member from V2 to V4.1, including R1, compared on the axes that actually changed.", datum: "config" },
  { slug: "architecture", n: "02", part: "I", title: "The block, in three generations", short: "Architecture", desc: "MLA, DeepSeekMoE, hash layers and compressed attention, with tensor shapes.", datum: "config" },
  { slug: "budget", n: "03", part: "I", title: "Where 1.6 trillion parameters live", short: "Parameter budget", desc: "Total versus active, expert versus attention, and the arithmetic of sparsity.", datum: "config" },
  { slug: "cache", n: "04", part: "I", title: "The compression of memory", short: "KV cache", desc: "From a 576-wide latent to 1M tokens: what each generation stores, and why.", datum: "config" },
  { slug: "frontier", n: "05", part: "I", title: "Attention at one million tokens", short: "Long context", desc: "DSA, CSA and HCA compared by the arithmetic each one refuses to pay.", datum: "config" },

  { slug: "lens", n: "06", part: "II", title: "The answer, forming in layers", short: "Logit lens", desc: "Intermediate states read through the final unembedding, layer by layer.", datum: "measured" },
  { slug: "commitment", n: "07", part: "II", title: "When a prediction becomes inevitable", short: "Commitment", desc: "The layer at which each answer first becomes the argmax, across the corpus.", datum: "measured" },
  { slug: "anatomy", n: "08", part: "II", title: "What each block writes", short: "Layer anatomy", desc: "Attention and MoE writes measured separately in the residual stream.", datum: "measured" },
  { slug: "routing", n: "09", part: "II", title: "The router's 64-way decision", short: "Routing", desc: "Every token's top-k experts, the entropy of the choice, and its margin.", datum: "measured" },
  { slug: "experts", n: "10", part: "II", title: "A cartography of specialists", short: "Experts", desc: "Which experts fire for which computations, and which fire together.", datum: "measured" },
  { slug: "attention", n: "11", part: "II", title: "Sixteen readers of a latent", short: "Attention", desc: "Head taxonomy, distance decay, sinks and induction in MLA.", datum: "measured" },
  { slug: "latent", n: "12", part: "II", title: "The 512-dimensional memory", short: "KV latent", desc: "The compressed vector itself: rank, variance and geometry across depth.", datum: "measured" },
  { slug: "ablation", n: "13", part: "II", title: "What happens when you remove it", short: "Ablation", desc: "Zero a block, a shared expert or a routed expert; measure the damage.", datum: "measured" },
  { slug: "circuit", n: "14", part: "II", title: "The wiring behind a prediction", short: "Circuit", desc: "Forward wiring and direct target attributions, with the caveat that neither is yet a circuit.", datum: "measured" },
  { slug: "geometry", n: "15", part: "II", title: "The shape of the stream", short: "Geometry", desc: "CKA between layers, effective rank, and the trajectory of a thought.", datum: "measured" },
  { slug: "probes", n: "16", part: "II", title: "What becomes linearly readable", short: "Probing", desc: "Six properties, one linear probe per layer, grouped cross-validation.", datum: "measured" },
  { slug: "context", n: "17", part: "II", title: "Retrieval at four thousand tokens", short: "Retrieval", desc: "Needle retrieval measured to 4k, then the cache arithmetic to 1M.", datum: "mixed" },

  { slug: "mechanism", n: "18", part: "III", title: "What counts as a mechanism?", short: "Mechanism", desc: "Components, paths, and the interventions that separate a story from a causal claim.", datum: "mixed" },
  { slug: "acdc", n: "19", part: "III", title: "Finding the circuit, edge by edge", short: "ACDC", desc: "Automated circuit discovery: the loop, its assumptions, and an offline readout pilot.", datum: "mixed" },
  { slug: "synthesis", n: "20", part: "III", title: "One architecture, three substitutions", short: "Synthesis", desc: "The family thesis, graded claim by claim, with what the atlas does not show.", datum: "mixed" },
  { slug: "methods", n: "21", part: "III", title: "How the numbers were made", short: "Methods", desc: "Evidence classes, the capture pipeline, the corpus, and the limits.", datum: "method" },

  { slug: "causal-ablation", n: "22", part: "IV", title: "The model, minus a head", short: "Causal ablation", desc: "Real head- and expert-level zero ablations on V2-Lite, measured with forward passes.", datum: "measured" },
  { slug: "edge-circuit", n: "23", part: "IV", title: "A circuit, ablated for real", short: "Edge circuit", desc: "A greedy edge search that reruns the model after every intervention, with a held-out validation prompt.", datum: "measured" },

  { slug: "transformer", n: "24", part: "V", title: "The transformer, from token to logit", short: "Transformer", desc: "The whole stack: tokenisation, residual stream, attention, MLP, norm and unembedding.", datum: "method" },
  { slug: "understanding", n: "25", part: "V", title: "How meaning travels", short: "Meaning & context", desc: "What research shows about semantics, verbs, sarcasm and reasoning, and how long context is carried.", datum: "mixed" },
  { slug: "theory-attention", n: "26", part: "V", title: "Attention is a weighted read", short: "Attention", desc: "Queries, keys and values: how one token decides what to read from the sequence.", datum: "method" },
  { slug: "theory-self-attention", n: "27", part: "V", title: "A sequence reading itself", short: "Self-attention", desc: "The causal mask, the current token and the difference between attention and recurrence.", datum: "method" },
  { slug: "theory-multi-head", n: "28", part: "V", title: "Many heads, one residual", short: "Multi-head", desc: "Parallel subspaces, concatenation and the output projection.", datum: "method" },
  { slug: "theory-moe", n: "29", part: "V", title: "A large memory, sparsely read", short: "Mixture of experts", desc: "Routers, top-k selection, shared experts and why active parameters differ from stored parameters.", datum: "method" },
  { slug: "theory-sparse-attention", n: "30", part: "V", title: "Reading less of the context", short: "Sparse attention", desc: "DSA indexing, CSA/HCA compression and the sliding window as three ways to avoid a full matrix.", datum: "method" },
  { slug: "theory-latent", n: "31", part: "V", title: "The cache as a latent", short: "Compressed latent", desc: "MLA stores a shared latent instead of per-head key/value tensors.", datum: "method" },
  { slug: "theory-ced", n: "32", part: "V", title: "An encoder for the context", short: "CED", desc: "Causal encoder–decoder: the V4.1 split that computes a full cache only where it is needed.", datum: "method" },
  { slug: "theory-reasoning", n: "33", part: "V", title: "Reasoning is a post-train", short: "Reasoning post-training", desc: "Why R1 is a training branch on V3, not a new architecture.", datum: "method" },
  { slug: "keywords", n: "34", part: "V", title: "The vocabulary of the atlas", short: "Keyword atlas", desc: "The terms that recur across the report, each with a one-glance explanation.", datum: "method" },
  { slug: "member-blocks", n: "35", part: "V", title: "Every member, as a block", short: "Member blocks", desc: "The same architecture diagram instantiated for each DeepSeek family member.", datum: "config" },
];

export const PARTS = [
  { id: "I", label: "The family", sub: "config arithmetic" },
  { id: "II", label: "The depth", sub: "measured on V2-Lite" },
  { id: "III", label: "The account", sub: "synthesis & method" },
  { id: "IV", label: "Causal follow-up", sub: "real interventions" },
  { id: "V", label: "Theory", sub: "visual primer" },
];

export function useRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("?")[0];
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => {
      setRoute(read());
      window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function href(slug: string) {
  return `#/${slug}`;
}

export function useMeasure<T extends HTMLElement>() {
  const [w, setW] = useState(760);
  const observer = useRef<ResizeObserver | null>(null);
  // A callback ref, not a mount effect: measured nodes often appear only
  // after their JSON resolves, so there is no element on first render.
  const ref = useCallback((el: T | null) => {
    observer.current?.disconnect();
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    observer.current = ro;
    setW(el.clientWidth);
  }, []);
  useEffect(() => () => observer.current?.disconnect(), []);
  return { ref, width: w };
}

/* ------------------------------------------------------------------ utils */
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function fmt(v: number, d = 2): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(d);
}

export function pct(v: number, d = 1): string {
  return `${(v * 100).toFixed(d)}%`;
}

export function compact(v: number, d = 1): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e12) return `${(v / 1e12).toFixed(d)}T`;
  if (a >= 1e9) return `${(v / 1e9).toFixed(d)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(d)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(d)}k`;
  return v.toFixed(0);
}

export function bytes(v: number): string {
  if (v >= 2 ** 30) return `${(v / 2 ** 30).toFixed(2)} GiB`;
  if (v >= 2 ** 20) return `${(v / 2 ** 20).toFixed(1)} MiB`;
  if (v >= 2 ** 10) return `${(v / 2 ** 10).toFixed(1)} KiB`;
  return `${v.toFixed(0)} B`;
}

export function count(v: number): string {
  return v.toLocaleString();
}

export function mix(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(lerp(v, pb[i], clamp(t, 0, 1))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function ramp(t: number, stops: [number, string][]): string {
  const x = clamp(t, 0, 1);
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (x >= p0 && x <= p1) return mix(c0, c1, (x - p0) / (p1 - p0 || 1));
  }
  return stops[stops.length - 1][1];
}

/** Sequential: paper → blue → near-black. */
export const SEQ: [number, string][] = [
  [0, "#f3f1ea"],
  [0.18, "#dfe4f8"],
  [0.42, "#a9b6ee"],
  [0.66, "#5f73df"],
  [0.85, "#2c46be"],
  [1, "#101a4d"],
];

/** Diverging: teal ← paper → ember. */
export const DIV: [number, string][] = [
  [0, "#17656b"],
  [0.42, "#dfe8e4"],
  [0.5, "#f7f5ef"],
  [0.58, "#f0dbcd"],
  [1, "#b8461c"],
];

export const INK = {
  blue: "#2c46be",
  signal: "#2b46f0",
  ember: "#b8461c",
  teal: "#17656b",
  olive: "#5c6b2f",
  plum: "#6a3f73",
  slate: "#4b5563",
  sand: "#b58a4a",
};

export function tokenLabel(t: string): string {
  return t.replace(/Ġ/g, "␣").replace(/Ċ/g, "⏎").replace(/^##/, "").replace(/\n/g, "⏎");
}

export function bare(t: string): string {
  return tokenLabel(t).replace(/[␣⏎]/g, "").trim().toLowerCase();
}

export function usePrompt() {
  const [, force] = useState(0);
  const [id, setId] = useState<string | null>(null);
  const choose = (next: string) => {
    setId(next);
    force((n) => n + 1);
  };
  return [id, choose] as const;
}

export function quantile(vals: number[], q: number): number {
  if (!vals.length) return 0;
  const s = [...vals].sort((a, b) => a - b);
  return s[clamp(Math.floor(q * (s.length - 1)), 0, s.length - 1)];
}

export function mean(v: number[]) {
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}

export function sum(v: number[]) {
  return v.reduce((a, b) => a + b, 0);
}
