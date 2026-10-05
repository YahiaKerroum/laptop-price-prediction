"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { Combo, Roll, Segmented, spring } from "./ui";
import { type Catalog, type Prediction, type Similar, cpuShort, da, featureLabel, k, n, title } from "@/lib/data";

const RAM = [4, 8, 12, 16, 32, 64];
const SSD = [0, 128, 256, 512, 1024, 2048];
const CONDITIONS = [
  { value: null, label: "Not sure" },
  { value: 1, label: "Fair" },
  { value: 2, label: "Good" },
  { value: 3, label: "Never used" },
];
const SCREENS = [11.6, 12.5, 13.3, 14, 15.6, 16, 17.3];
const RESOLUTIONS = [
  { value: 1, label: "HD (1366×768)" },
  { value: 3, label: "Full HD (1920×1080)" },
  { value: 5, label: "QHD / 2.5K" },
  { value: 8, label: "4K" },
];

type Form = {
  cpu: string | null;
  gpu: string | null;
  ram: number;
  ssd: number;
  brand: string;
  condition: number | null;
  city: string;
  screen: number | null;
  resolution: number | null;
};

export default function Estimator({ catalog, marketMedian }: { catalog: Catalog; marketMedian: number }) {
  const [f, setF] = useState<Form>({
    cpu: "Intel Core i5-1135G7 @ 2.40GHz",
    gpu: null,
    ram: 8,
    ssd: 512,
    brand: "THINKPAD",
    condition: 2,
    city: "",
    screen: null,
    resolution: null,
  });
  const [more, setMore] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [similar, setSimilar] = useState<Similar[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = <K extends keyof Form>(key: K, v: Form[K]) => setF((s) => ({ ...s, [key]: v }));

  const cpuItems = useMemo(
    () => catalog.cpus.map((c) => ({ value: c.name, label: cpuShort(c.name), meta: c.cpu_mark ? `${k(c.cpu_mark)} pts` : undefined })),
    [catalog.cpus],
  );
  const gpuItems = useMemo(
    () => catalog.gpus.map((g) => ({ value: g.name, label: g.name, meta: `${n(g.listings)} ads` })),
    [catalog.gpus],
  );

  // Live pricing: debounce edits, cancel stale requests.
  useEffect(() => {
    const body = {
      cpu_name: f.cpu ?? undefined,
      gpu_name: f.gpu ?? undefined,
      ram_gb: f.ram,
      ssd_gb: f.ssd,
      brand: f.brand || "UNKNOWN",
      condition: f.condition ?? undefined,
      city: f.city || "UNKNOWN",
      screen_size: f.screen ?? undefined,
      resolution: f.resolution ?? undefined,
      listing_year: new Date().getFullYear() > 2025 ? 2025 : new Date().getFullYear(),
    };
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        const opts = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal };
        const [p, s] = await Promise.all([fetch("/api/predict?explain=true", opts), fetch("/api/similar?k=5", opts)]);
        if (!p.ok) throw new Error(p.status === 503 ? "model" : `HTTP ${p.status}`);
        setResult(await p.json());
        setSimilar(s.ok ? await s.json() : []);
        setError(null);
      } catch (e) {
        if ((e as Error).name !== "AbortError") setError((e as Error).message);
      } finally {
        if (!ctrl.signal.aborted) setBusy(false);
      }
    }, 280);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [f]);

  return (
    <div className="est">
      <form className="form" onSubmit={(e) => e.preventDefault()} aria-label="Laptop specs">
        <div className="field cpu-field">
          <label htmlFor="cpu">
            Processor <span className="hint">The detail that matters most</span>
          </label>
          <Combo id="cpu" items={cpuItems} value={f.cpu} onChange={(v) => set("cpu", v)} placeholder="Search, e.g. i5-1135G7 or Ryzen 5 5500U" />
        </div>

        <div className="field">
          <span className="label">Memory</span>
          <Segmented label="Memory" options={RAM.map((v) => ({ value: v, label: `${v} GB` }))} value={f.ram} onChange={(v) => set("ram", v)} />
        </div>

        <div className="field">
          <span className="label">SSD</span>
          <Segmented
            label="SSD"
            options={SSD.map((v) => ({ value: v, label: v === 0 ? "None" : v >= 1024 ? `${v / 1024} TB` : `${v} GB` }))}
            value={f.ssd}
            onChange={(v) => set("ssd", v)}
          />
        </div>

        <div className="field">
          <span className="label">Condition</span>
          <Segmented label="Condition" options={CONDITIONS} value={f.condition} onChange={(v) => set("condition", v)} />
        </div>

        <div className="two-col">
          <div className="field">
            <label htmlFor="brand">Model line</label>
            <select id="brand" className="select" value={f.brand} onChange={(e) => set("brand", e.target.value)}>
              <option value="">Other / not sure</option>
              {catalog.brands.map((b) => (
                <option key={b} value={b}>
                  {title(b)}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="gpu">
              Graphics card <span className="hint">if any</span>
            </label>
            <Combo id="gpu" items={gpuItems} value={f.gpu} onChange={(v) => set("gpu", v)} placeholder="Built-in" />
          </div>
        </div>

        <button type="button" className="more" aria-expanded={more} onClick={() => setMore((m) => !m)}>
          {more ? "Fewer details" : "Screen and city"}
          <motion.svg width="12" height="8" viewBox="0 0 12 8" animate={{ rotate: more ? 180 : 0 }} transition={spring}>
            <path d="M1 1.5l5 5 5-5" stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round" />
          </motion.svg>
        </button>
        <AnimatePresence initial={false}>
          {more && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={spring}
              style={{ overflow: "hidden" }}
            >
              <div className="two-col" style={{ paddingTop: 2 }}>
                <div className="field">
                  <label htmlFor="screen">Screen size</label>
                  <select id="screen" className="select" value={f.screen ?? ""} onChange={(e) => set("screen", e.target.value ? Number(e.target.value) : null)}>
                    <option value="">Not sure</option>
                    {SCREENS.map((s) => (
                      <option key={s} value={s}>
                        {s}&Prime;
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="res">Resolution</label>
                  <select id="res" className="select" value={f.resolution ?? ""} onChange={(e) => set("resolution", e.target.value ? Number(e.target.value) : null)}>
                    <option value="">Not sure</option>
                    {RESOLUTIONS.map((r) => (
                      <option key={r.value} value={r.value}>
                        {r.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field" style={{ gridColumn: "1 / -1" }}>
                  <label htmlFor="city">City</label>
                  <select id="city" className="select" value={f.city} onChange={(e) => set("city", e.target.value)}>
                    <option value="">Anywhere</option>
                    {catalog.cities.map((c) => (
                      <option key={c} value={c}>
                        {title(c)}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </form>

      <Result result={result} similar={similar} busy={busy} error={error} marketMedian={marketMedian} />
    </div>
  );
}

function Result({ result, similar, busy, error, marketMedian }: { result: Prediction | null; similar: Similar[]; busy: boolean; error: string | null; marketMedian: number }) {
  const range = result?.range_dzd ?? (result ? [result.estimate_dzd, result.estimate_dzd] : null);
  const wide = result?.wide_range_dzd ?? range;

  // Fixed-ish scale so the band moves rather than the axis.
  const lo = Math.min(wide?.[0] ?? 0, marketMedian) * 0.85;
  const hi = Math.max(wide?.[1] ?? 1, marketMedian) * 1.06;
  const at = (v: number) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))}%`;
  const span = (a: number, b: number) => `${((b - a) / (hi - lo)) * 100}%`;

  const factors = (result?.contributions ?? []).slice(0, 6).map((c) => ({ label: featureLabel(c.feature), pct: Math.expm1(c.contribution) * 100 }));
  const maxF = Math.max(10, ...factors.map((x) => Math.abs(x.pct)));

  return (
    <aside className="result" aria-live="polite">
      <div className="k">
        <span className="live" data-busy={busy} aria-hidden="true" />
        Likely asking price
      </div>

      {range ? (
        <>
          <div className="range-big">
            <Roll value={range[0]} format={da} /> – <Roll value={range[1]} format={da} />
            <span className="cur">DA</span>
          </div>
          <div className="range-sub">Half of comparable ads ask a price in this range.</div>

          <div className="band" aria-hidden="true">
            <div className="band-track" />
            {wide && <motion.div className="band-wide" animate={{ left: at(wide[0]), width: span(wide[0], wide[1]) }} transition={spring} />}
            <motion.div className="band-likely" animate={{ left: at(range[0]), width: span(range[0], range[1]) }} transition={spring} />
            <motion.div className="band-pin" animate={{ left: at(result!.estimate_dzd) }} transition={spring}>
              <b>
                <Roll value={result!.estimate_dzd} format={da} />
              </b>
              <i />
            </motion.div>
            <motion.div className="band-ref" animate={{ left: at(marketMedian) }} transition={spring}>
              <span>Market median {k(marketMedian)}</span>
            </motion.div>
          </div>

          <div className="stats">
            <div>
              <label>Most likely</label>
              <b>
                <Roll value={result!.estimate_dzd} format={da} />
              </b>
            </div>
            <div>
              <label>Quick sale</label>
              <b>
                <Roll value={range[0]} format={da} />
              </b>
            </div>
            <div>
              <label>Optimistic</label>
              <b>
                <Roll value={range[1]} format={da} />
              </b>
            </div>
          </div>

          {result?.precision === "low" && (
            <div className="nudge">Pick your processor to narrow this. Naming it roughly halves the range.</div>
          )}
        </>
      ) : (
        <div className="range-big placeholder">— · —</div>
      )}

      {error && (
        <div className="err">
          {error === "model"
            ? "The pricing service is running but has no trained model. Run "
            : "The pricing service isn’t reachable. Start it with "}
          <code>{error === "model" ? "make train" : "make serve"}</code> and this updates on its own.
        </div>
      )}

      {factors.length > 0 && (
        <div className="why">
          <h4>What moved the price</h4>
          {factors.map((x) => (
            <div className="factor" key={x.label}>
              <span>{x.label}</span>
              <div className="bar">
                <motion.i
                  initial={false}
                  animate={{
                    left: x.pct >= 0 ? "50%" : `${50 - (Math.abs(x.pct) / maxF) * 50}%`,
                    width: `${(Math.abs(x.pct) / maxF) * 50}%`,
                    background: x.pct >= 0 ? "#3e9bff" : "#ff6b5e",
                  }}
                  transition={spring}
                />
              </div>
              <span className="v num" style={{ color: x.pct >= 0 ? "#9cc9ff" : "#ffa59c" }}>
                {x.pct >= 0 ? "+" : "−"}
                {Math.abs(x.pct).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {similar.length > 0 && (
        <div className="similar">
          <h4>Real ads like it</h4>
          {similar.slice(0, 4).map((s) => (
            <div className="sim-row" key={s.rank}>
              <div>
                <b>{s.brand && s.brand !== "UNKNOWN" && s.brand !== "OTHER" ? title(s.brand) : "Laptop"}</b>
                <div className="small">
                  {[s.ram_gb && `${s.ram_gb} GB`, s.ssd_gb ? `${s.ssd_gb >= 1000 ? Math.round(s.ssd_gb / 1000) + " TB" : s.ssd_gb + " GB"} SSD` : null, s.city && s.city !== "UNKNOWN" && s.city !== "OTHER" ? title(s.city) : null, s.listing_year]
                    .filter(Boolean)
                    .join(", ")}
                </div>
              </div>
              <b className="num">{da(s.asking_price_dzd)} DA</b>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
