"use client";

import { motion, useReducedMotion, useScroll, useSpring, useTransform } from "motion/react";
import { useRef } from "react";
import { da } from "@/lib/data";

/** The demo machine on the hero screen: a real, common configuration. */
const DEMO = { low: 67600, mid: 77700, high: 89200, scaleLo: 40000, scaleHi: 130000 };
const pct = (v: number) => ((v - DEMO.scaleLo) / (DEMO.scaleHi - DEMO.scaleLo)) * 100;

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const p = useSpring(scrollYProgress, { stiffness: 140, damping: 28, mass: 0.6 });
  const prog = reduce ? scrollYProgress : p;

  const lid = useTransform(prog, [0, 0.34], reduce ? [0, 0] : [-86, 0]);
  const lidOpacity = useTransform(prog, [0.01, 0.09], reduce ? [1, 1] : [0, 1]);
  const glow = useTransform(prog, [0.22, 0.4], [0, 1]);
  const copyY = useTransform(prog, [0, 0.4], [0, -50]);
  const leadFade = useTransform(prog, [0.04, 0.18], [1, 0]);
  const copyFade = useTransform(prog, [0.55, 0.8], [1, 0.0]);
  const band = useTransform(prog, [0.34, 0.6], [0, 1]);
  const bandLeft = useTransform(band, [0, 1], [pct(DEMO.mid), pct(DEMO.low)]);
  const bandWidth = useTransform(band, [0, 1], [0, pct(DEMO.high) - pct(DEMO.low)]);
  const left = useTransform(bandLeft, (v) => `${v}%`);
  const width = useTransform(bandWidth, (v) => `${v}%`);
  const price = useTransform(band, (v) => `${da(DEMO.low * v)} – ${da(DEMO.high * v)}`);
  const pinOpacity = useTransform(prog, [0.52, 0.6], [0, 1]);
  const bars = useTransform(prog, [0.4, 0.66], [0, 1]);
  const hint = useTransform(prog, [0, 0.08], [1, 0]);
  const ghostScale = useTransform(prog, [0, 1], [1, 1.12]);
  const ghostY = useTransform(prog, [0, 1], [0, -60]);
  const halo = useTransform(prog, [0.18, 0.45], [0, 1]);
  const haloScale = useTransform(prog, [0.18, 0.5], [0.6, 1]);
  const floor = useTransform(prog, [0.02, 0.34], [0.55, 1]);

  const chip = (from: number, x: number) => ({
    opacity: useTransform(prog, [from, from + 0.1], [0, 1]),
    y: useTransform(prog, [from, from + 0.14], [30, 0]),
    x: useTransform(prog, [from, from + 0.14], [x, 0]),
  });
  /* eslint-disable react-hooks/rules-of-hooks */
  const c1 = chip(0.46, -30);
  const c2 = chip(0.52, 30);
  const c3 = chip(0.58, 30);
  /* eslint-enable react-hooks/rules-of-hooks */

  const heights = [38, 56, 72, 88, 100, 82, 60, 44, 30];

  return (
    <section ref={ref} className="hero" id="top" aria-label="Introduction">
      <div className="hero-stick">
        <motion.div className="ghost" aria-hidden="true" style={{ scale: ghostScale, y: ghostY }}>
          قيمة
        </motion.div>

        <motion.div className="hero-copy" style={{ y: copyY, opacity: copyFade }}>
          <h1 className="display">
            What&rsquo;s your
            <br />
            laptop worth?
          </h1>
          <motion.p className="lead" style={{ opacity: leadFade }}>
            Fair asking prices for used laptops in Algeria, learned from 16,000 real listings.
          </motion.p>
        </motion.div>

        <div className="stage">
          <motion.div className="halo" aria-hidden="true" style={{ opacity: halo, scale: haloScale }} />
          <motion.div className="floor" aria-hidden="true" style={{ scaleX: floor }} />
          <motion.div className="lid" style={{ rotateX: lid, opacity: lidOpacity }}>
            <div className="lid-back" aria-hidden="true"><b>قيمة</b></div>
            <div className="lid-shell">
              <div className="screen">
                <div className="notch" />
                <motion.div className="screen-ui" style={{ opacity: glow }}>
                  <div style={{ fontSize: "clamp(.55rem,1.4vw,.85rem)", color: "#6e6e73", fontWeight: 600 }}>
                    ThinkPad · Core i5-1135G7 · 8 GB · 512 GB
                  </div>
                  <motion.div
                    className="num"
                    style={{ fontSize: "clamp(1.1rem,3.6vw,2.4rem)", fontWeight: 700, letterSpacing: "-.045em", marginTop: ".3em" }}
                  >
                    {price}
                  </motion.div>
                  <div style={{ fontSize: "clamp(.5rem,1.2vw,.75rem)", color: "#86868b" }}>DA · likely asking price</div>
                  <div style={{ position: "relative", height: "8%", marginTop: "8%" }}>
                    <div style={{ position: "absolute", inset: "35% 0", borderRadius: 99, background: "#e3e3e8" }} />
                    <motion.div
                      style={{ position: "absolute", top: "10%", bottom: "10%", left, width, borderRadius: 99, background: "linear-gradient(90deg,#0a5bd3,#3e9bff)" }}
                    />
                    <motion.div
                      style={{ position: "absolute", top: "-20%", bottom: "-20%", left: `${pct(DEMO.mid)}%`, aspectRatio: "1", transform: "translateX(-50%)", borderRadius: "50%", background: "#fff", boxShadow: "0 1px 6px rgba(0,0,0,.3)", opacity: pinOpacity }}
                    />
                  </div>
                  <div style={{ flex: 1, display: "flex", alignItems: "flex-end", gap: "2.4%", marginTop: "6%" }}>
                    {heights.map((h, i) => (
                      <motion.div
                        key={i}
                        style={{
                          flex: 1,
                          height: `${h}%`,
                          borderRadius: "6px 6px 2px 2px",
                          background: i === 4 ? "#0a5bd3" : i === 3 || i === 5 ? "#8fc1f7" : "#dbe8f8",
                          transformOrigin: "50% 100%",
                          scaleY: bars,
                        }}
                      />
                    ))}
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.div>
          <div className="base" />

          <motion.div className="chip" style={{ left: "-13%", top: "46%", ...c1 }}>
            <small>Processor</small>Core i5-1135G7
          </motion.div>
          <motion.div className="chip" style={{ right: "-9%", top: "10%", ...c2 }}>
            <small>Condition</small>Good
          </motion.div>
          <motion.div className="chip" style={{ right: "-6%", top: "64%", ...c3 }}>
            <small>Half of similar ads</small>±14% of the middle
          </motion.div>
        </div>

        <motion.div className="scroll-hint small" style={{ opacity: hint }} aria-hidden="true">
          Scroll to open
        </motion.div>
      </div>
    </section>
  );
}
