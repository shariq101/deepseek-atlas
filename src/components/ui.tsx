import React, { ReactNode } from "react";
import { CHAPTERS, Chapter, href, tokenLabel } from "../lib/atlas";

/* ------------------------------------------------------------------ chrome */
export function ChapterHead({
  n, part, kicker, title, standfirst, chips,
}: {
  n: string; part?: string; kicker: string; title: ReactNode; standfirst: ReactNode; chips?: ReactNode;
}) {
  return (
    <header className="chapter-head">
      <div className="kicker">
        <span>
          {part ? `${part} · ` : ""}{n} — {kicker}
        </span>
      </div>
      <h1 className="title">{title}</h1>
      <p className="standfirst">{standfirst}</p>
      {chips && <div className="byline">{chips}</div>}
    </header>
  );
}

export function Panel({
  title, tools, children, note, tight, className = "", style,
}: {
  title?: ReactNode; tools?: ReactNode; children: ReactNode; note?: ReactNode;
  tight?: boolean; className?: string; style?: React.CSSProperties;
}) {
  return (
    <figure className={"panel " + className} style={style}>
      {(title || tools) && (
        <div className="panel-head">
          <div className="panel-title">{title}</div>
          <div className="spacer" />
          {tools}
        </div>
      )}
      <div className={"panel-body" + (tight ? " tight" : "")}>{children}</div>
      {note && <figcaption className="panel-note">{note}</figcaption>}
    </figure>
  );
}

export function Caption({ children }: { children: ReactNode }) {
  return <p className="caption">{children}</p>;
}

export function Method({ label = "How this is measured", children }: { label?: string; children: ReactNode }) {
  return (
    <div className="method">
      <div className="label">{label}</div>
      {children}
    </div>
  );
}

export function Provenance({
  kind, children,
}: {
  kind: "config" | "measured" | "model" | "method" | "mixed"; children: ReactNode;
}) {
  return (
    <span className={"prov " + kind}>
      <span className="prov-dot" />
      {children}
    </span>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <div className="prose">{children}</div>;
}

export function Stat({ k, v, s }: { k: string; v: ReactNode; s?: ReactNode }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {s && <div className="s">{s}</div>}
    </div>
  );
}

export function Chip({ children, tone = "" }: { children: ReactNode; tone?: "" | "blue" | "ember" | "teal" }) {
  return <span className={"chip " + tone}>{children}</span>;
}

export function Seg<T extends string>({
  value, options, onChange,
}: {
  value: T; options: { v: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.v} className={o.v === value ? "on" : ""} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { c: string; label: string; line?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend-item" key={it.label}>
          <span className={"swatch" + (it.line ? " line" : "")} style={{ background: it.c }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function Colorbar({
  stops, min, max, label, width = 150, format,
}: {
  stops: [number, string][]; min: number; max: number; label?: string;
  width?: number; format?: (v: number) => string;
}) {
  const grad = `linear-gradient(90deg, ${stops.map(([p, c]) => `${c} ${p * 100}%`).join(", ")})`;
  return (
    <div className="ramp">
      <span>{format ? format(min) : min}</span>
      <span className="bar" style={{ background: grad, width }} />
      <span>{format ? format(max) : max}</span>
      {label && <span className="ramp-label">{label}</span>}
    </div>
  );
}

export function Tok({ t, style, title }: { t: string; style?: React.CSSProperties; title?: string }) {
  return (
    <span className="token" style={style} title={title ?? t}>
      {tokenLabel(t) || "∅"}
    </span>
  );
}

export function PageFoot({ index }: { index: number }) {
  const prev = CHAPTERS[index - 1];
  const next = CHAPTERS[index + 1];
  return (
    <nav className="pagefoot">
      {prev ? (
        <a href={href(prev.slug)}>
          <span className="lbl">Previous</span>
          <span>{prev.n} · {prev.title}</span>
        </a>
      ) : <span />}
      {next ? (
        <a href={href(next.slug)} className="next">
          <span className="lbl">Next</span>
          <span>{next.n} · {next.title}</span>
        </a>
      ) : <span />}
    </nav>
  );
}

export function Footnotes({ items }: { items: (string | ReactNode)[] }) {
  return (
    <div className="footnotes">
      <div className="fn-label">Notes &amp; provenance</div>
      <ol>
        {items.map((t, i) => (
          <li key={i}>{typeof t === "string" ? <span dangerouslySetInnerHTML={{ __html: t }} /> : t}</li>
        ))}
      </ol>
    </div>
  );
}

export function DataState({
  loading, error, what, children,
}: {
  loading: boolean; error: string | null; what: string; children: ReactNode;
}) {
  if (loading)
    return (
      <div className="data-state">
        <span className="pulse" />
        Reading {what}…
      </div>
    );
  if (error)
    return (
      <div className="data-state err">
        <b>{what} is not in the workspace.</b> {error}. The capture kernel writes it to
        <code> public/data/</code>; see <a href={href("methods")}>Methods</a>.
      </div>
    );
  return <>{children}</>;
}

export function ChapterTag({ datum }: { datum: Chapter["datum"] }) {
  const label = datum === "config" ? "config-derived" : datum === "measured" ? "measured" : datum === "method" ? "method" : "mixed provenance";
  return <Provenance kind={datum === "method" ? "method" : datum}>{label}</Provenance>;
}
