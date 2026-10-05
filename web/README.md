# Qima — web front end

Next.js site for the laptop price model. Market charts, deals and the parts
catalog are static JSON in `public/data/`; live estimates go to the FastAPI
service through the `/api/*` proxy in `next.config.ts`.

```bash
# 1. the API (from the repo root)
make serve                      # or: uvicorn laptop_price.api.main:app --port 8000

# 2. the site
cd web && npm install && npm run dev     # http://localhost:3000
```

Set `API_URL` if the API is not on `http://127.0.0.1:8000`.

After retraining the model or rebuilding `features.csv`, refresh the static data:

```bash
python scripts/export_web_data.py
```
