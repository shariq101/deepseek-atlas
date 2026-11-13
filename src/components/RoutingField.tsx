import React, { useEffect, useMemo, useRef, useState } from "react";
import { Architecture, Meta, RouterEntry, clamp, fmt, mix, useJson, useMeasure } from "../lib/atlas";

type Cell = { l: number; e: number; v: number };

/**
 * The hero instrument. With measured router data it is a 27 × 64 field of
 * expert load: each dot is one expert in one layer, area is its share of that
 * layer's routed tokens. Without the capture it falls back to the checkpoint's
 * layer plan, which is config-derived, and says so.
 */
export default function RoutingField() {
  const metaQ = useJson<Meta>("meta");
  const routerQ = useJson<Record<string, RouterEntry>>("router");
  const archQ = useJson<Architecture>("architecture");
  const meta = metaQ.data?.meta ?? null;
  const router = routerQ.data?.router ?? null;
  const arch = archQ.data?.architecture ?? null;
  const { ref, width } = useMeasure<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const [ready, setReady] = useState(false);

  const matrix = useMemo(() => {
    if (!router || !meta) return null;
    const ids = Object.keys(router);
    const L = meta.layers;
    const E = meta.experts;
    const m = Array.from({ length: L }, () => new Array(E).fill(0));
    for (const id of ids) {
      const r = router[id];
      for (let l = 0; l < L; l++) {
        const row = r.load?.[String(l)] ?? r.load?.[l as unknown as string] ?? [];
        for (let e = 0; e < E; e++) m[l][e] += row[e] ?? 0;
      }
    }
    const norm = m.map((row) => {
      const hi = Math.max(...row, 1);
      return row.map((v) => v / hi);
    });
    return { m: norm, L, E, prompts: ids.length };
  }, [router, meta]);

  const plan = useMemo(() => arch?.focus?.v2lite?.plan ?? null, [arch]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const h = canvas.clientHeight || 330;
    canvas.width = width * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, h);

    const pad = { l: 42, r: 18, t: 26, b: 30 };
    const innerW = width - pad.l - pad.r;
    const innerH = h - pad.t - pad.b;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const start = performance.now();
    let raf = 0;

    const draw = (now: number) => {
      const t = reduce ? 1 : clamp((now - start) / 1100, 0, 1);
      ctx.clearRect(0, 0, width, h);

      // labels
      ctx.fillStyle = "#82868e";
      ctx.font = "8.5px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("experts", pad.l - 9, pad.t - 9);
      ctx.textAlign = "left";
      ctx.fillText("0", pad.l - 25, pad.t + 3);
      ctx.fillText("63", pad.l - 25, pad.t + innerH + 2);
      ctx.textAlign = "left";
      ctx.fillText("depth →", pad.l, h - 8);

      if (matrix) {
        const { L, E } = matrix;
        const cw = innerW / L;
        const rh = innerH / E;
        const sweep = reduce ? 2 : ((now - start) / 5200) % 1.35 - 0.15;
        for (let l = 0; l < L; l++) {
          const alpha = clamp(t * 2 - (l / L) * 1.1, 0, 1);
          if (alpha <= 0) continue;
          for (let e = 0; e < E; e++) {
            const v = matrix.m[l][e];
            if (v < 0.008) continue;
            const x = pad.l + (l + 0.5) * cw;
            const y = pad.t + (e + 0.5) * rh;
            const near = Math.abs(l / L - sweep) < 0.06;
            const r = 0.8 + Math.sqrt(v) * Math.min(3.2, cw * 0.33);
            const col = mix("#2b46f0", "#6a3f73", (e / E) * 0.75);
            ctx.globalAlpha = alpha * (0.22 + 0.78 * Math.sqrt(v)) * (near ? 1 : 0.82);
            ctx.fillStyle = near ? "#b8461c" : col;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fill();
          }
          // layer tick
          if (l % 3 === 0) {
            ctx.globalAlpha = 0.7;
            ctx.fillStyle = "#82868e";
            ctx.font = "8px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText(String(l), pad.l + (l + 0.5) * cw, h - 18);
          }
        }
        ctx.globalAlpha = 0.12;
        ctx.strokeStyle = "#101217";
        ctx.lineWidth = 0.5;
        for (let l = 0; l <= L; l += 3) {
          ctx.beginPath();
          ctx.moveTo(pad.l + l * cw, pad.t);
          ctx.lineTo(pad.l + l * cw, pad.t + innerH);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      } else if (plan) {
        const L = plan.length;
        const cw = innerW / L;
        const rows = [
          { key: "attn", y: 0.22, label: "attention" },
          { key: "ffn", y: 0.52, label: "FFN" },
          { key: "compress", y: 0.8, label: "KV" },
        ];
        for (let l = 0; l < L; l++) {
          const p = plan[l];
          const alpha = clamp(t * 2 - (l / L) * 1.1, 0, 1);
          if (alpha <= 0) continue;
          rows.forEach((row, ri) => {
            const y = pad.t + row.y * innerH;
            const v = ri === 0 ? (p.attn === "mla" ? 0.55 : 0.3)
              : ri === 1 ? (p.ffn === "moe" ? 0.9 : 0.35)
              : (p.compress > 1 ? Math.min(1, p.compress / 128) : 0.2);
            const r = 1.2 + v * Math.min(4, cw * 0.38);
            ctx.globalAlpha = alpha * 0.75;
            ctx.fillStyle = ri === 1 ? "#2b46f0" : ri === 0 ? "#6a3f73" : "#17656b";
            ctx.beginPath();
            ctx.arc(pad.l + (l + 0.5) * cw, y, r, 0, Math.PI * 2);
            ctx.fill();
          });
          if (l % 3 === 0) {
            ctx.globalAlpha = 0.7;
            ctx.fillStyle = "#82868e";
            ctx.font = "8px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText(String(l), pad.l + (l + 0.5) * cw, h - 18);
          }
        }
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = "#82868e";
        ctx.textAlign = "left";
        ctx.font = "8.5px 'JetBrains Mono', monospace";
        rows.forEach((row) => ctx.fillText(row.label, pad.l, pad.t + row.y * innerH - 10));
        ctx.globalAlpha = 1;
      }
      if (!reduce && t < 1) raf = requestAnimationFrame(draw);
      else setReady(true);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [width, matrix, plan]);

  const onMove = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    if (!matrix) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const pad = { l: 42, r: 18, t: 26, b: 30 };
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const innerW = width - pad.l - pad.r;
    const innerH = rect.height - pad.t - pad.b;
    const l = Math.floor(((x - pad.l) / innerW) * matrix.L);
    const e = Math.floor(((y - pad.t) / innerH) * matrix.E);
    if (l >= 0 && l < matrix.L && e >= 0 && e < matrix.E) setHover({ l, e, v: matrix.m[l][e] });
    else setHover(null);
  };

  return (
    <div ref={ref}>
      <div className="hero-strip">
        <span className="viz-label">
          {matrix
            ? `expert load · ${matrix.prompts} prompts · ${matrix.L} layers × ${matrix.E} experts`
            : "layer plan · V2-Lite · attention / FFN / KV compression"}
        </span>
        <span className="legend-inline">
          {matrix ? "area = routed-token share · colour = expert" : "config-derived · capture pending"}
        </span>
      </div>
      <canvas ref={canvasRef} className="hero-canvas" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      <div className="hero-legend">
        <span>
          {hover && matrix
            ? `layer ${hover.l} · expert ${hover.e} · ${fmt(hover.v * 100, 1)}% of layer max`
            : "hover the field to inspect one expert in one layer"}
        </span>
        {!matrix && <span>the measured router field appears when the capture lands</span>}
        {ready && <span className="ok-dot">●</span>}
      </div>
    </div>
  );
}
