"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";
import Emblem from "./Emblem";
import { spring } from "./ui";

const TABS = [
  { id: "estimate", label: "Estimate" },
  { id: "deals", label: "Deals" },
  { id: "market", label: "Market" },
];

export default function Nav() {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && setActive(e.target.id)),
      { rootMargin: "-45% 0px -50% 0px" },
    );
    ["top", ...TABS.map((t) => t.id)].forEach((id) => {
      const el = document.getElementById(id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  return (
    <nav className="nav" aria-label="Sections">
      <a className="mark" href="#top" aria-label="Qima, back to top">
        <Emblem size={30} />
        <span>Qima</span>
      </a>
      {TABS.map((t) => (
        <a key={t.id} className="tab" href={`#${t.id}`} data-active={active === t.id}>
          {active === t.id && <motion.span layoutId="nav-pill" className="pill" transition={spring} />}
          {t.label}
        </a>
      ))}
    </nav>
  );
}
