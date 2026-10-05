"use client";

import { motion, useInView, useReducedMotion } from "motion/react";
import { useId, useRef } from "react";

/**
 * Qima's mark: a laptop that opens like a scallop shell, holding a pearl with a
 * processor inside it. The processor is the part that moves a laptop's price most.
 * Vector redraw of public/qima-logo.png so it stays sharp and can animate.
 */
const FAN =
  "M34.97 37.58 L57.75 34.38 C62.65 32.15 59.39 23.06 54.19 24.44 C57.79 20.44 51.19 13.4 46.97 16.74 C48.69 11.64 39.83 7.8 37.28 12.54 C36.83 7.18 27.17 7.18 26.72 12.54 C24.17 7.8 15.31 11.64 17.03 16.74 C12.81 13.4 6.21 20.44 9.81 24.44 C4.61 23.06 1.35 32.15 6.25 34.38 L29.03 37.58 Z";
const RIBS =
  "M37.12 34.87 L53.96 24.58 M35.45 33.09 L46.82 16.95 M33.22 32.12 L37.23 12.8 M30.78 32.12 L26.77 12.8 M28.55 33.09 L17.18 16.95 M26.88 34.87 L10.04 24.58";
const BOWL = "M4 40 A28 5 0 0 0 60 40 C60 51.5 47 58 32 58 C17 58 4 51.5 4 40 Z";
const PINS =
  "M30 32.5v-1.6 M32 32.5v-1.6 M34 32.5v-1.6 M30 40.5v1.6 M32 40.5v1.6 M34 40.5v1.6 M28 34.5h-1.6 M28 36.5h-1.6 M28 38.5h-1.6 M36 34.5h1.6 M36 36.5h1.6 M36 38.5h1.6";

export default function Emblem({ size = 28, animate = false, className }: { size?: number; animate?: boolean; className?: string }) {
  const id = useId().replace(/:/g, "");
  const ref = useRef<SVGSVGElement>(null);
  const seen = useInView(ref, { once: true, margin: "-80px" });
  const reduce = useReducedMotion();
  const play = animate && !reduce;
  const open = !play || seen;
  const ease = [0.22, 1, 0.36, 1] as const;
  const u = (name: string) => `url(#${name}${id})`;

  return (
    <svg ref={ref} width={size} height={size} viewBox="0 0 64 64" className={className} role="img" aria-label="Qima">
      <defs>
        <linearGradient id={`fan${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#a9d1fb" />
          <stop offset="1" stopColor="#3d8ae6" />
        </linearGradient>
        <linearGradient id={`bowl${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#56565c" />
          <stop offset="1" stopColor="#1d1d21" />
        </linearGradient>
        <linearGradient id={`rim${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#e2e5ea" />
          <stop offset="1" stopColor="#9da2ac" />
        </linearGradient>
        <linearGradient id={`inner${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1e1e22" />
          <stop offset="1" stopColor="#3b3b40" />
        </linearGradient>
        <radialGradient id={`pearl${id}`} cx="0.38" cy="0.3" r="0.78">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.6" stopColor="#eef2f8" />
          <stop offset="1" stopColor="#b9c5d6" />
        </radialGradient>
        <radialGradient id={`glow${id}`}>
          <stop offset="0" stopColor="#3e9bff" stopOpacity="0.5" />
          <stop offset="1" stopColor="#3e9bff" stopOpacity="0" />
        </radialGradient>
      </defs>

      {animate && (
        <motion.circle
          cx="32"
          cy="36"
          r="20"
          fill={u("glow")}
          initial={play ? { opacity: 0, scale: 0.4 } : false}
          animate={open ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 1.2, ease, delay: 0.5 }}
          style={{ transformOrigin: "32px 36px" }}
        />
      )}

      {/* lid: the scallop fan, hinged at the back of the bowl */}
      <motion.g
        initial={play ? { scaleY: 0.1 } : false}
        animate={open ? { scaleY: 1 } : {}}
        transition={{ duration: 1.1, ease }}
        style={{ transformOrigin: "32px 38px" }}
      >
        <path d={FAN} fill={u("fan")} stroke="#1c6ad6" strokeWidth="1.5" strokeLinejoin="round" />
        <path d={RIBS} stroke="#fff" strokeOpacity="0.7" strokeWidth="0.9" strokeLinecap="round" fill="none" />
      </motion.g>

      {/* base: a deep graphite bowl with a silver rim */}
      <path d={BOWL} fill={u("bowl")} />
      <ellipse cx="32" cy="40" rx="28" ry="5" fill={u("rim")} />
      <ellipse cx="32" cy="40.4" rx="25.8" ry="3.9" fill={u("inner")} />

      {/* the pearl, with a processor in it */}
      <motion.g
        initial={play ? { opacity: 0, scale: 0.5, y: 4 } : false}
        animate={open ? { opacity: 1, scale: 1, y: 0 } : {}}
        transition={{ duration: 0.9, ease, delay: 0.35 }}
        style={{ transformOrigin: "32px 40px" }}
      >
        <ellipse cx="32" cy="43.6" rx="7.5" ry="1.5" fill="#000" fillOpacity="0.45" />
        <circle cx="32" cy="36.5" r="9.3" fill={u("pearl")} />
        <path d={PINS} stroke="#0a5bd3" strokeWidth="1" strokeLinecap="round" />
        <rect x="28" y="32.5" width="8" height="8" rx="1.4" fill="#f4f8fe" stroke="#0a5bd3" strokeWidth="1.4" />
        <rect x="30.3" y="34.8" width="3.4" height="3.4" rx="0.6" fill="#fff" stroke="#0a5bd3" strokeOpacity="0.5" strokeWidth="0.6" />
        <ellipse cx="28.2" cy="31.4" rx="2.1" ry="1.3" fill="#fff" transform="rotate(-30 28.2 31.4)" />
      </motion.g>
    </svg>
  );
}
