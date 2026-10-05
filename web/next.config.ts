import type { NextConfig } from "next";

// The pricing API (FastAPI, `make serve`). Proxied so the browser never needs CORS
// and the URL can change per deployment without rebuilding client code.
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
