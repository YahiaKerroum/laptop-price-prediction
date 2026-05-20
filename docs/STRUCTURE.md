Project structure proposal for laptop-price-prediction

Overview
- Keep root minimal: README.md plus project metadata.

Directories
- data/
  - raw/      : original source CSVs (never overwrite)
  - processed/: cleaned and derived datasets
  - mappings/ : lookup tables and mapping CSVs
- notebooks/
  - cleaning/           : cleaning and EDA notebooks
  - feature-engineering/: feature construction notebooks
  - association/        : association-rule or specialized analysis
- scripts/ : small reusable scripts (non-notebook)
- docs/    : documentation and migration guides

File mappings (suggested)
- data/raw/
  - data.csv
  - cpus.csv
  - gpus.csv
  - cpu_prices.csv
  - gpu_prices.csv

- data/processed/
  - data_cleaned.csv
  - data_with_cpus.csv
  - data_with_cpus_gpus.csv
  - cleanedramstoragedata.csv

- data/mappings/
  - cpu_ddr_map.csv
  - cpu_storage_map.csv

- notebooks/cleaning/
  - clean_price.ipynb
  - clean_ram_storage.ipynb
  - clean_screen_related.ipynb

- notebooks/feature-engineering/
  - cpus_gpus_handling.ipynb
  - preProcessing.ipynb

- notebooks/association/
  - association_rules.ipynb

- scripts/
  - add_missing_cpus_to_maps.py

Guidelines
- Raw data: never modify files in `data/raw/`; create derived copies in `data/processed/`.
- Naming: use lowercase and hyphens or underscores consistently.
- Notebooks: keep exploratory work in notebooks; production scripts go in `scripts/` or `src/`.
- Documentation: update `docs/STRUCTURE.md` when structure changes.
