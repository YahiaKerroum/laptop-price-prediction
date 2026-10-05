"use client";

import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import { Segmented, spring } from "./ui";
import { type Deal, cpuShort, da, n, title } from "@/lib/data";

const SORTS = [
  { value: "discount", label: "Biggest discount" },
  { value: "savings", label: "Most saved" },
  { value: "price", label: "Cheapest" },
] as const;
type Sort = (typeof SORTS)[number]["value"];

export default function Deals({ deals }: { deals: Deal[] }) {
  const brands = useMemo(() => {
    const counts = new Map<string, number>();
    deals.forEach((d) => d.brand && counts.set(d.brand, (counts.get(d.brand) ?? 0) + 1));
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([b]) => b);
  }, [deals]);
  const maxPrice = useMemo(() => Math.ceil(Math.max(...deals.map((d) => d.price)) / 10000) * 10000, [deals]);

  const [brand, setBrand] = useState("");
  const [cap, setCap] = useState(maxPrice);
  const [sort, setSort] = useState<Sort>("discount");
  const [shown, setShown] = useState(9);

  const list = useMemo(() => {
    const rows = deals.filter((d) => (!brand || d.brand === brand) && d.price <= cap);
    const key = { discount: (d: Deal) => -d.discount, savings: (d: Deal) => -(d.expected - d.price), price: (d: Deal) => d.price }[sort];
    return rows.sort((a, b) => key(a) - key(b));
  }, [deals, brand, cap, sort]);

  return (
    <>
      <div className="filters">
        <div className="field">
          <label htmlFor="deal-brand">Model line</label>
          <select id="deal-brand" className="select" value={brand} onChange={(e) => { setBrand(e.target.value); setShown(9); }}>
            <option value="">All</option>
            {brands.map((b) => (
              <option key={b} value={b}>{title(b)}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="deal-cap">
            Up to <span className="num" style={{ fontWeight: 650 }}>{da(cap)} DA</span>
          </label>
          <input id="deal-cap" type="range" min={20000} max={maxPrice} step={5000} value={cap} onChange={(e) => setCap(Number(e.target.value))} />
        </div>
        <div className="field" style={{ flex: 1.6 }}>
          <span className="label">Sort</span>
          <Segmented label="Sort deals" options={[...SORTS]} value={sort} onChange={setSort} />
        </div>
      </div>

      <p className="small" style={{ margin: "0 0 1rem" }}>
        {n(list.length)} listings priced well below what comparable laptops ask. Listings that look like bait were removed first.
      </p>

      <motion.div className="deal-grid" layout>
        <AnimatePresence mode="popLayout">
          {list.slice(0, shown).map((d) => (
            <motion.article
              key={`${d.price}-${d.expected}-${d.cpu}-${d.city}-${d.ram}`}
              className="deal"
              layout
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.94 }}
              transition={spring}
              whileHover={{ y: -6 }}
            >
              <div className="top">
                <span className="brand">{d.brand ? title(d.brand) : "Laptop"}</span>
                <span className="off num">{Math.round(d.discount)}% under</span>
              </div>
              <div className="price num">
                {da(d.price)}
                <small>DA</small>
              </div>
              <div className="was num">
                <s>{da(d.expected)} DA</s>
                <span>Save {da(d.expected - d.price)} DA</span>
              </div>
              <div className="meter" aria-hidden="true">
                <motion.i initial={{ width: "100%" }} animate={{ width: `${(d.price / d.expected) * 100}%` }} transition={{ ...spring, delay: 0.15 }} />
              </div>
              {d.cpu && <div className="cpu">{cpuShort(d.cpu)}{d.gpu ? ` + ${d.gpu}` : ""}</div>}
              <div className="small">
                {[d.ram && `${d.ram} GB`, d.ssd ? `${d.ssd >= 1000 ? Math.round(d.ssd / 1000) + " TB" : d.ssd + " GB"} SSD` : null, d.condition, d.city && title(d.city), d.year]
                  .filter(Boolean)
                  .join(", ")}
              </div>
            </motion.article>
          ))}
        </AnimatePresence>
      </motion.div>

      {list.length === 0 && (
        <p className="body" style={{ textAlign: "center", padding: "3rem 0" }}>
          No deals under {da(cap)} DA for this model line. Raise the price limit or choose all model lines.
        </p>
      )}
      {list.length > shown && (
        <button className="show-more" onClick={() => setShown((s) => s + 9)}>
          Show more deals
        </button>
      )}
    </>
  );
}
