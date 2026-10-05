"use client";

import { motion } from "motion/react";
import { CountUp } from "./ui";
import { type Market as M, da, k, n, title } from "@/lib/data";

const view = { once: true, margin: "-80px" } as const;
const ease = [0.22, 1, 0.36, 1] as const;

function Story({ head, body, children }: { head: string; body: string; children: React.ReactNode }) {
  return (
    <div className="story">
      <motion.div initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={view} transition={{ duration: 0.8, ease }}>
        <h3 className="h3">{head}</h3>
        <p className="body">{body}</p>
      </motion.div>
      <div className="chart">{children}</div>
    </div>
  );
}

function Bars({ rows, max, color = "var(--blue)" }: { rows: { label: string; value: number; note: string; hot?: boolean }[]; max: number; color?: string }) {
  return (
    <div>
      {rows.map((r, i) => (
        <div className="hbar" key={r.label}>
          <span className="t">{r.label}</span>
          <motion.div
            className="b"
            style={{ width: `${(r.value / max) * 100}%`, background: r.hot === false ? "#c9d9ef" : color }}
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={view}
            transition={{ duration: 0.9, ease, delay: i * 0.05 }}
          />
          <span className="v num">{r.note}</span>
        </div>
      ))}
    </div>
  );
}

function Scatter({ points }: { points: M["scatter"] }) {
  const W = 640, H = 380, pad = { l: 46, r: 10, t: 10, b: 34 };
  const xMax = 32000, yMax = 400000;
  const x = (v: number) => pad.l + (Math.min(v, xMax) / xMax) * (W - pad.l - pad.r);
  const y = (v: number) => H - pad.b - (Math.min(v, yMax) / yMax) * (H - pad.t - pad.b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Asking price against processor benchmark score; each dot is one listing">
      {[0, 100000, 200000, 300000, 400000].map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#e1e1e6" />
          <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" fontSize="12" fill="#86868b">{t ? k(t) : "0"}</text>
        </g>
      ))}
      {[0, 10000, 20000, 30000].map((t) => (
        <text key={t} x={x(t)} y={H - 10} textAnchor="middle" fontSize="12" fill="#86868b">{t ? `${t / 1000}k pts` : "0"}</text>
      ))}
      <motion.g initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={view} transition={{ duration: 1.2 }}>
        {points.map(([c, p, g], i) => (
          <circle key={i} cx={x(c)} cy={y(p)} r={g ? 3 : 2.4} fill={g ? "#0a5bd3" : "#7fb3f5"} fillOpacity={g ? 0.6 : 0.4} />
        ))}
      </motion.g>
    </svg>
  );
}

export default function Market({ m }: { m: M }) {
  const [b1, b2] = m.brands;
  const topTwo = Math.round(b1.share + b2.share);

  const tp = m.heat.rows.indexOf(b1.brand);
  const fair = m.heat.values[tp]?.[0] ?? null;
  const fresh = m.heat.values[tp]?.[m.heat.cols.length - 1] ?? null;
  const condRatio = fair && fresh ? fresh / fair : null;

  const r8 = m.ram.find((r) => r.gb === 8);
  const r16 = m.ram.find((r) => r.gb === 16);
  const r16share = r16 ? Math.round((r16.count / m.ram.reduce((s, r) => s + r.count, 0)) * 100) : 0;

  const cities = m.cities;
  const top = cities[0], bottom = cities[cities.length - 1];

  const heatMax = Math.max(...m.heat.values.flat().filter((v): v is number => v != null));

  return (
    <>
      <div className="kpis">
        {([
          [<CountUp key="a" value={m.listings} format={(v) => n(Math.round(v))} />, "real listings"],
          [<CountUp key="b" value={m.median} format={(v) => `${k(v)} DA`} />, "median asking price"],
          [<CountUp key="c" value={m.p75} format={(v) => `${k(Math.min(v, m.p25))}–${k(v)}`} />, "where the middle half sits"],
          [<CountUp key="d" value={m.citiesTracked} format={(v) => String(Math.round(v))} />, "cities and districts"],
        ] as [React.ReactNode, string][]).map(([v, l], i) => (
          <motion.div className="kpi" key={l} initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={view} transition={{ duration: 0.7, ease, delay: i * 0.06 }}>
            <b>{v}</b>
            <span>{l}</span>
          </motion.div>
        ))}
      </div>

      <Story
        head={`${title(b1.brand)} and ${title(b2.brand)} make up ${topTwo}% of the market.`}
        body={`Ex-corporate business laptops dominate Algerian listings. They ask about ${k(b1.median)} DA on median, while a MacBook asks closer to ${k(m.brands.find((b) => b.brand === "MACBOOK")?.median ?? m.median)}.`}
      >
        <Bars rows={m.brands.map((b) => ({ label: title(b.brand), value: b.share, note: `${b.share.toFixed(1)}%` }))} max={m.brands[0].share} />
      </Story>

      {condRatio && (
        <Story
          head={`A never-used ${title(b1.brand)} asks ${condRatio.toFixed(1)}× a worn one.`}
          body="Condition moves the price more than the badge on the lid. Median asking price by model line and stated condition."
        >
          <div className="heat" style={{ gridTemplateColumns: `6.6rem repeat(${m.heat.cols.length}, 1fr)` }}>
            <span />
            {m.heat.cols.map((c) => <span className="ch" key={c}>{c}</span>)}
            {m.heat.rows.map((r, ri) => (
              <div key={r} style={{ display: "contents" }}>
                <span className="rh">{title(r)}</span>
                {m.heat.values[ri].map((v, ci) => {
                  const t = v ? v / heatMax : 0;
                  return (
                    <motion.span
                      key={ci}
                      className="cell num"
                      style={{ background: `color-mix(in srgb, #0a5bd3 ${Math.round(8 + t * 92)}%, #fff)`, color: t > 0.42 ? "#fff" : "#1d1d1f" }}
                      initial={{ opacity: 0, scale: 0.9 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      viewport={view}
                      transition={{ duration: 0.5, ease, delay: (ri + ci) * 0.03 }}
                    >
                      {v ? k(v) : "—"}
                    </motion.span>
                  );
                })}
              </div>
            ))}
          </div>
        </Story>
      )}

      {r8 && r16 && (
        <Story
          head={`16 GB is the norm now: ${r16share}% of listings.`}
          body={`Going from 8 to 16 GB lifts the median ask from ${da(r8.median)} to ${da(r16.median)} DA. Memory tracks the rest of the machine, so treat this as a tier, not an upgrade price.`}
        >
          <Bars
            rows={m.ram.map((r) => ({ label: `${r.gb} GB`, value: r.median, note: `${k(r.median)}`, hot: r.gb === 16 }))}
            max={Math.max(...m.ram.map((r) => r.median))}
          />
        </Story>
      )}

      <Story
        head="A faster processor raises the ceiling, not the floor."
        body="Every dot is a listing. Fast machines can ask a lot, yet plenty of them still sell cheap. Darker dots have a dedicated graphics card."
      >
        <Scatter points={m.scatter} />
      </Story>

      <Story
        head={`In ${title(top.city)}, the median ad asks ${(top.median / bottom.median).toFixed(1)}× what it does in ${title(bottom.city)}.`}
        body={`Where you list matters. Median asking price in every area with at least 100 listings; ${title(top.city)} is highlighted.`}
      >
        <Bars
          rows={cities.slice(0, 12).map((c, i) => ({ label: title(c.city), value: c.median, note: k(c.median), hot: i === 0 }))}
          max={top.median}
        />
      </Story>
    </>
  );
}
