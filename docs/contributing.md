# Contributing

## Setup

Everything runs in Docker; the only host requirement is Docker itself.

```bash
make build
make test
```

## Before opening a PR

```bash
make lint     # ruff + black --check
make test     # 135 tests
make verify   # repo still matches the original project folder
```

`make format` applies ruff fixes and black.

## Where code goes

| If you are changing… | Put it in… |
|---|---|
| A cleaning rule (parsing, normalisation, a threshold) | `src/laptop_price/cleaning/` |
| A feature | `src/laptop_price/features/build.py` + the declared tuples |
| A metric, baseline or split | `src/laptop_price/evaluation/` |
| Model architecture or the artifact format | `src/laptop_price/models/` |
| A magic number | `src/laptop_price/config.yaml` — never inline |
| Narrative, charts, exploration | `notebooks/` |

**Notebooks import from `src/`; they do not define pipeline logic.** The original project put
the same normalisation function in three notebooks with three slightly different bodies. If
you find yourself writing a function in a notebook that another notebook also needs, it
belongs in the package.

## Conventions

- Line length 100; ruff and black enforce it.
- Type hints on public functions.
- Docstrings say **why**, not what. `# add 1 to x` is noise; "the scraper emits
  space-separated timestamps, which some pandas versions parse to NaT" is not.
- New columns must be added to `NUMERIC_FEATURES` or `CATEGORICAL_FEATURES` **and** to the
  pandera schema. A feature that exists but is not declared will not reach the model.

## Testing

Every pure function in `cleaning/` gets tests. These used to be notebook cells whose only
verification was a `print` statement, and two genuine bugs were found by writing tests for
them.

Two cases are worth writing explicitly:

- **The obvious one.** `parse_ram_size("16GB") == 16`.
- **The one that bit us.** `cpu_mark` arriving as `"19,108"`; `needs_swap` on two storage
  values; an unbounded baseline producing R² of −48,000.

The artifact round-trip test (fit → save → load → predict) guards the exact failure this
project previously shipped. Do not weaken it.

## Data

- **Never write to `data/raw/`.** Derived data goes in `interim/` or `processed/`.
- Changing a cleaning rule changes `features.csv`. Re-run `make pipeline && make train` and
  say what moved in the PR description.
- Lookup tables belong in `data/mappings/` as CSV, not as dict literals in a cell. The
  126-entry CPU correction table was moved out for exactly this reason.

## Reporting numbers

Quote the **time-based** split as the headline. The random-split number is higher and is kept
only for comparability with the original reports.

Never hand-type a metric into a document. `docs/modeling.md` quotes what training printed,
and the model card is generated from `metadata.json`. The original reports drifted from their
own data — citing row counts that did not match the split that ran — and that is the failure
mode this avoids.

If a change makes the model worse, report that. A PR saying "MedAPE went from 12.2% to 12.8%,
here is why the tradeoff is worth it" is fine. One quietly reporting the better of two splits
is not.
