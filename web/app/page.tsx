import Deals from "@/components/Deals";
import Emblem from "@/components/Emblem";
import Estimator from "@/components/Estimator";
import Hero from "@/components/Hero";
import Market from "@/components/Market";
import Nav from "@/components/Nav";
import Ticker from "@/components/Ticker";
import type { Catalog, Deal, Market as MarketData } from "@/lib/data";
import catalogJson from "@/public/data/catalog.json";
import dealsJson from "@/public/data/deals.json";
import marketJson from "@/public/data/market.json";

const catalog = catalogJson as Catalog;
const deals = dealsJson as Deal[];
const market = marketJson as unknown as MarketData;

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Ticker deals={deals} />

        <section className="block" id="estimate" aria-labelledby="estimate-h">
          <div className="wrap">
            <div className="sec-head">
              <h2 className="h2" id="estimate-h">Price yours in seconds.</h2>
              <p className="lead">
                Name the processor and the price tightens right away. Everything else is optional, and a blank is better than a guess.
              </p>
            </div>
            <Estimator catalog={catalog} marketMedian={market.median} />
          </div>
        </section>

        <section className="block" id="deals" aria-labelledby="deals-h">
          <div className="wrap">
            <div className="sec-head">
              <h2 className="h2" id="deals-h">Priced below the market.</h2>
              <p className="lead">Real ads asking far less than comparable laptops, with likely scams filtered out.</p>
            </div>
            <Deals deals={deals} />
          </div>
        </section>

        <section className="block" id="market" aria-labelledby="market-h">
          <div className="wrap">
            <div className="sec-head">
              <h2 className="h2" id="market-h">How the market prices.</h2>
              <p className="lead">
                What {market.listings.toLocaleString("fr-FR")} listings from {market.yearSpan[0]} to {market.yearSpan[1]} say about used laptops in Algeria.
              </p>
            </div>
            <Market m={market} />
          </div>
        </section>
      </main>

      <footer className="foot">
        <div className="wrap">
          <div className="foot-brand">
            <Emblem size={180} animate />
            <div>
              <div className="foot-word">
                Qima <span lang="ar">قيمة</span>
              </div>
              <p className="lead">Every laptop has something valuable inside. We help you put a fair price on it.</p>
            </div>
          </div>
          <div className="foot-row small">
            <span>Qima estimates asking prices from listings, not final sale prices.</span>
            <span>Prices in Algerian dinars (DA).</span>
          </div>
        </div>
      </footer>
    </>
  );
}
