export type Cpu = { name: string; listings: number; cpu_mark: number | null; cores: number | null; family: string | null; igpu: string | null };
export type Gpu = { name: string; listings: number; g3d_mark: number | null };
export type Catalog = { cpus: Cpu[]; gpus: Gpu[]; brands: string[]; cities: string[] };

export type Deal = {
  price: number; expected: number; discount: number;
  brand: string | null; city: string | null; ram: number | null; ssd: number | null;
  cpuMark: number | null; gpuMark: number | null; condition: string | null; year: number | null;
  cpu: string | null; gpu: string | null;
};

export type Market = {
  listings: number; median: number; p25: number; p75: number;
  brandsTracked: number; citiesTracked: number; yearSpan: [number, number];
  brands: { brand: string; share: number; count: number; median: number }[];
  years: { year: number; median: number; count: number }[];
  cities: { city: string; median: number; count: number }[];
  scatter: [number, number, number][];
  heat: { rows: string[]; cols: string[]; values: (number | null)[][] };
  ram: { gb: number; median: number; count: number }[];
};

export type Prediction = {
  estimate_dzd: number;
  range_dzd: [number, number] | null;
  wide_range_dzd: [number, number] | null;
  precision: "high" | "low" | null;
  caveat: string;
  contributions: { feature: string; contribution: number }[] | null;
};

export type Similar = {
  rank: number; asking_price_dzd: number; brand: string | null; city: string | null;
  ram_gb: number | null; ssd_gb: number | null; cpu_mark: number | null;
  cpu_family: string | null; listing_year: number | null;
};

const fmt = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
/** Algerian convention: space-grouped, "DA". */
export const da = (v: number) => fmt.format(Math.round(v / 100) * 100);
export const n = (v: number) => fmt.format(v);
export const k = (v: number) => `${Math.round(v / 1000)}k`;

const PROPER: Record<string, string> = {
  THINKPAD: "ThinkPad", MACBOOK: "MacBook", ELITEBOOK: "EliteBook", PROBOOK: "ProBook", VIVOBOOK: "VivoBook",
  IDEAPAD: "IdeaPad", ZENBOOK: "ZenBook", THINKBOOK: "ThinkBook", ROG: "ROG", TUF: "TUF", MSI: "MSI",
  ZBOOK: "ZBook", EXPERTBOOK: "ExpertBook", OMEN: "OMEN", XPS: "XPS",
};
export const title = (s: string | null | undefined) =>
  PROPER[(s ?? "").toUpperCase()] ?? (s ?? "").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

/** "Intel Core i5-1135G7 @ 2.40GHz" -> "Intel Core i5-1135G7" */
export const cpuShort = (s: string | null | undefined) => (s ?? "").replace(/\s*@.*$/, "");

const LABELS: Record<string, string> = {
  cpu_mark: "Processor speed",
  gpu_g3d_mark: "Graphics power",
  gpu_g2d_mark: "Graphics (2D)",
  gpu_tdp: "Graphics wattage",
  tdp: "Processor wattage",
  gpu_to_cpu_ratio: "Graphics vs processor",
  cpu_generation_normalized: "Processor generation",
  SSD_SIZE: "SSD size",
  RAM_SIZE: "Memory",
  RAM_TYPE: "Memory type",
  HDD_SIZE: "Hard drive",
  spec_Etat: "Condition",
  etat_is_missing: "Condition not stated",
  listing_month_index: "Listing date",
  listing_year: "Listing year",
  listing_month: "Listing month",
  month_sin: "Season",
  month_cos: "Season",
  estimated_component_cost: "Parts value",
  perf_per_expected_dinar: "Speed for the money",
  ppi: "Screen sharpness",
  pixels: "Resolution",
  SCREEN_SIZE_SNAPPED: "Screen size",
  SCREEN_RESOLUTION_ENC: "Resolution",
  storage_per_ram: "Storage vs memory",
  total_storage: "Total storage",
  total_tdp: "Total wattage",
  cores: "Processor cores",
  has_dedicated_gpu: "Dedicated graphics",
  model_family: "Performance tier",
};
const PREFIXES: [string, string][] = [
  ["cpu_family_", "Processor"],
  ["cpu_manufacturer_", "Processor maker"],
  ["brand_", "Model"],
  ["city_grouped_", "City"],
];

export function featureLabel(raw: string): string {
  const f = raw.replace(/^(num|cat|remainder)__/, "");
  if (LABELS[f]) return LABELS[f];
  for (const [p, label] of PREFIXES) {
    if (f.startsWith(p)) return `${label}: ${title(f.slice(p.length).replace(/_/g, " "))}`;
  }
  return title(f.replace(/_/g, " "));
}
