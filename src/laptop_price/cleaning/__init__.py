"""Normalisation of the scraped listing fields.

Each module owns one family of columns and exposes pure, testable functions plus
one ``clean_*`` entry point that takes and returns a DataFrame. The notebooks and
the CLI both call the same functions, so there is exactly one implementation of
each rule.

Scope note
----------
The hand-curated CPU/GPU enrichment that produces
``data/processed/pre_processed_data.csv`` from the raw scrape lives in
``notebooks/01_preprocessing.ipynb``. Its correction tables were lifted out to
``data/mappings/`` so they are inspectable data; ``cleaning.cpu`` carries the
normalisation and lookup that *serving* needs at inference time.
"""

from laptop_price.cleaning import cpu, memory, price, screen

__all__ = ["cpu", "memory", "price", "screen"]
