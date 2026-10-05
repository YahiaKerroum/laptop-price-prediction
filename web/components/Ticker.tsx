import { type Deal, cpuShort, da, title } from "@/lib/data";

/** A market board of real listings: what they ask against what similar laptops ask. */
export default function Ticker({ deals }: { deals: Deal[] }) {
  const rows = deals.filter((d) => d.brand && d.cpu).slice(0, 18);
  const item = (d: Deal, i: number) => (
    <li key={i}>
      <b>{title(d.brand)}</b>
      <span className="t-cpu">{cpuShort(d.cpu)}</span>
      <span className="num">{da(d.price)} DA</span>
      <em className="num">▼ {Math.round(d.discount)}%</em>
    </li>
  );
  return (
    <div className="ticker" aria-label="Recent listings priced below the market">
      <ul>
        {rows.map(item)}
        {/* second copy makes the loop seamless; hidden from assistive tech */}
        {rows.map((d, i) => (
          <li key={`c${i}`} aria-hidden="true">
            <b>{title(d.brand)}</b>
            <span className="t-cpu">{cpuShort(d.cpu)}</span>
            <span className="num">{da(d.price)} DA</span>
            <em className="num">▼ {Math.round(d.discount)}%</em>
          </li>
        ))}
      </ul>
    </div>
  );
}
