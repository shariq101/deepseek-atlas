import React from "react";
import { CHAPTER_FILES, CHAPTERS, Meta, PARTS, href, useAvailability, useJson, useRoute } from "./lib/atlas";
import Overview from "./pages/Overview";
import Family from "./pages/Family";
import Architecture from "./pages/Architecture";
import Budget from "./pages/Budget";
import Cache from "./pages/Cache";
import Frontier from "./pages/Frontier";
import Lens from "./pages/Lens";
import Commitment from "./pages/Commitment";
import Anatomy from "./pages/Anatomy";
import Routing from "./pages/Routing";
import ExpertsPage from "./pages/Experts";
import Attention from "./pages/Attention";
import Latent from "./pages/Latent";
import Ablation from "./pages/Ablation";
import Circuit from "./pages/Circuit";
import GeometryPage from "./pages/Geometry";
import Probes from "./pages/Probes";
import ContextPage from "./pages/Context";
import Mechanism from "./pages/Mechanism";
import ACDC from "./pages/ACDC";
import Synthesis from "./pages/Synthesis";
import Methods from "./pages/Methods";
import CausalAblation from "./pages/CausalAblation";
import EdgeCircuitPage from "./pages/EdgeCircuit";
import UnderstandingPage from "./pages/Understanding";
import {
  KeywordsPage, TheoryAttentionPage, TheoryCedPage, TheoryLatentPage,
  TheoryMoePage, TheoryMultiHeadPage, TheoryReasoningPage, TheorySelfAttentionPage,
  TheorySparsePage, TransformerPage,
} from "./pages/Theory";
import FamilyBlocksPage from "./pages/FamilyBlocks";

class Boundary extends React.Component<
  { children: React.ReactNode; k: string },
  { err: Error | null }
> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) {
    return { err };
  }
  componentDidUpdate(prev: { k: string }) {
    if (prev.k !== this.props.k && this.state.err) this.setState({ err: null });
  }
  componentDidCatch(err: Error, info: React.ErrorInfo) {
    console.error("chapter render error", err, info.componentStack);
  }
  render() {
    if (this.state.err) {
      return (
        <section className="panel" style={{ marginTop: 40 }}>
          <div className="panel-head"><div className="panel-title">This chapter failed to render</div></div>
          <div className="panel-body">
            <pre className="error-pre">{String(this.state.err?.stack || this.state.err)}</pre>
          </div>
        </section>
      );
    }
    return this.props.children;
  }
}

const PAGES: Record<string, React.ReactNode> = {
  "": <Overview />,
  family: <Family />,
  architecture: <Architecture />,
  budget: <Budget />,
  cache: <Cache />,
  frontier: <Frontier />,
  lens: <Lens />,
  commitment: <Commitment />,
  anatomy: <Anatomy />,
  routing: <Routing />,
  experts: <ExpertsPage />,
  attention: <Attention />,
  latent: <Latent />,
  ablation: <Ablation />,
  circuit: <Circuit />,
  geometry: <GeometryPage />,
  probes: <Probes />,
  context: <ContextPage />,
  mechanism: <Mechanism />,
  acdc: <ACDC />,
  synthesis: <Synthesis />,
  methods: <Methods />,
  "causal-ablation": <CausalAblation />,
  "edge-circuit": <EdgeCircuitPage />,
  transformer: <TransformerPage />,
  understanding: <UnderstandingPage />,
  "theory-attention": <TheoryAttentionPage />,
  "theory-self-attention": <TheorySelfAttentionPage />,
  "theory-multi-head": <TheoryMultiHeadPage />,
  "theory-moe": <TheoryMoePage />,
  "theory-sparse-attention": <TheorySparsePage />,
  "theory-latent": <TheoryLatentPage />,
  "theory-ced": <TheoryCedPage />,
  "theory-reasoning": <TheoryReasoningPage />,
  keywords: <KeywordsPage />,
  "member-blocks": <FamilyBlocksPage />,
};

export default function App() {
  const route = useRoute();
  const idx = Math.max(0, CHAPTERS.findIndex((c) => c.slug === route));
  const chapter = CHAPTERS[idx] ?? CHAPTERS[0];
  const { data } = useJson<Meta>("meta");
  const meta = data?.meta;
  const avail = useAvailability();
  const needed = CHAPTER_FILES[chapter.slug] ?? [];
  const ready = avail.known && needed.length > 0 && needed.every((f) => avail.present.includes(f));
  const partial = avail.known && needed.some((f) => avail.present.includes(f));
  const totalFiles = avail.present.length + avail.missing.length;
  const provenance = !avail.known
    ? "checking provenance…"
    : avail.complete
      ? "config + measured · V2-Lite"
      : avail.measured
        ? `config + ${avail.present.length}/${totalFiles} measured`
        : "config data only · capture pending";
  const datumLabel = chapter.datum === "measured"
    ? (!avail.known ? "checking data" : ready ? "measured" : partial ? "measured · partial" : "capture pending")
    : chapter.datum === "mixed"
      ? (avail.measured ? "mixed" : "config + model")
      : chapter.datum === "method" ? "method" : "config arithmetic";

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName?.match(/INPUT|SELECT|TEXTAREA/)) return;
      if (e.key === "ArrowRight" && CHAPTERS[idx + 1]) window.location.hash = `/${CHAPTERS[idx + 1].slug}`;
      if (e.key === "ArrowLeft" && CHAPTERS[idx - 1]) window.location.hash = `/${CHAPTERS[idx - 1].slug}`;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [idx]);

  return (
    <div className="shell">
      <aside className="rail">
        <a className="brand" href={href("")}>
          <div className="mark">
            The DeepSeek<br /><em>Atlas</em>
          </div>
          <div className="sub">thirty-six chapters<br />for a sparse intelligence</div>
        </a>

        <div className="rail-meta">
          <div><b>2.4B → 49B</b> active</div>
          <div><b>6.4× → 32.5×</b> sparsity</div>
          <div
            className={avail.complete ? "ok" : "warn"}
            title={avail.missing.length ? `missing: ${avail.missing.join(", ")}` : "all measured artefacts present"}
          >
            ● {provenance}
          </div>
        </div>

        <nav className="rail-nav">
          {PARTS.map((part) => (
            <div key={part.id}>
              <div className="rail-group">
                <span>{part.id}</span> {part.label}
                <em>{part.sub}</em>
              </div>
              {CHAPTERS.filter((c) => c.part === part.id || (part.id === "I" && c.part === "Front")).map((c) => {
                const i = CHAPTERS.findIndex((x) => x.slug === c.slug);
                return (
                  <a key={c.slug || "overview"} href={href(c.slug)}
                    className={"rail-item" + (i === idx ? " active" : "")}>
                    <span className="idx">{c.n}</span>
                    <span>{c.short}</span>
                  </a>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="rail-foot">
          A measurement report, not a demo. Every figure names its instrument and
          its evidence class.
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <span className="crumb">
            {chapter.n} <span className="slash">/</span> {chapter.short}
          </span>
          <span className="spacer" />
          <span className="top-model">{meta?.modelName ?? "DeepSeek family"}</span>
          <span
            className={"datum-chip " + chapter.datum + (chapter.datum === "measured" && !ready ? " pending" : "")}
            title={needed.length ? `required: ${needed.join(", ")}${partial || ready ? "" : " · none present"}` : undefined}
          >
            {datumLabel}
          </span>
        </div>
        <div className="page">
          <Boundary k={route}>{PAGES[route] ?? PAGES[""]}</Boundary>
        </div>
      </main>
    </div>
  );
}
