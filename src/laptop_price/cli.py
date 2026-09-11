"""Command line entry points.

::

    python -m laptop_price pipeline     # pre_processed -> model_ready + data card
    python -m laptop_price train        # train, evaluate, write models/<version>/
    python -m laptop_price predict ...  # score one listing from the CLI
    python -m laptop_price deals        # rank the current underpriced listings

The Makefile wraps these so the container invocation stays short.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pandas as pd

from laptop_price import paths


def _print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def pipeline_main(argv: list[str] | None = None) -> int:
    """Build the model-ready matrix from the preprocessed listings."""
    parser = argparse.ArgumentParser(prog="laptop-pipeline", description=pipeline_main.__doc__)
    parser.add_argument("--no-validate", action="store_true", help="skip the pandera contract")
    parser.add_argument("--data-card", action="store_true", help="also write the data card CSV")
    args = parser.parse_args(argv)

    from laptop_price.data import build_model_ready, write_model_ready

    _print_header("Building the model-ready feature matrix")
    matrix = build_model_ready(validate=not args.no_validate)
    target = write_model_ready(matrix)

    print(f"rows                     {len(matrix):,}")
    print(f"features                 {matrix.shape[1] - 3}")
    print(f"unique spec signatures   {matrix['spec_signature'].nunique():,}")
    print(f"unit-ambiguous rows      {int(matrix['price_unit_ambiguous'].sum()):,}")
    print(f"written to               {target}")

    if args.data_card:
        from laptop_price.features.schema import describe_features

        card_path = paths.DOCS_DIR / "data-card.csv"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        describe_features(matrix).to_csv(card_path, index=False)
        print(f"data card                {card_path}")

    return 0


def train_main(argv: list[str] | None = None) -> int:
    """Train the price model and write a versioned artifact bundle."""
    parser = argparse.ArgumentParser(prog="laptop-train", description=train_main.__doc__)
    parser.add_argument("--no-quantiles", action="store_true", help="skip the interval models")
    parser.add_argument("--no-compare", action="store_true", help="skip Ridge/RandomForest")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the matrix first")
    args = parser.parse_args(argv)

    from laptop_price.data import build_model_ready, load_model_ready
    from laptop_price.models.registry import save_bundle
    from laptop_price.models.train import format_results, train
    from laptop_price.reporting import render_model_card

    if args.rebuild or not paths.FEATURES.exists():
        matrix = build_model_ready()
    else:
        matrix = load_model_ready()

    _print_header(f"Training on {len(matrix):,} listings")
    bundle, results = train(
        matrix,
        fit_quantiles=not args.no_quantiles,
        compare_models=not args.no_compare,
    )

    _print_header("Results")
    print(format_results(results))

    interval = bundle.metadata["metrics"].get("prediction_interval") or {}
    if interval:
        print(
            f"\nprediction interval: {interval['coverage_pct']:.1f}% coverage, "
            f"median width {interval['median_width']:,.0f} DZD "
            f"({interval['median_relative_width']:.0f}% of price)"
        )

    target = save_bundle(bundle, model_card=render_model_card(bundle))
    results.to_csv(target / "results.csv", index=False)
    print(f"\nartifacts written to {target}")
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    """Score a single listing described on the command line or as JSON."""
    parser = argparse.ArgumentParser(prog="laptop-predict", description=predict_main.__doc__)
    parser.add_argument("--json", help="listing as a JSON object, or '-' to read stdin")
    parser.add_argument("--ram", type=float, help="RAM in GB")
    parser.add_argument("--ssd", type=float, default=0, help="SSD in GB")
    parser.add_argument("--hdd", type=float, default=0, help="HDD in GB")
    parser.add_argument("--cpu-mark", type=float, help="PassMark CPU score")
    parser.add_argument("--gpu-mark", type=float, default=0, help="PassMark G3D score")
    parser.add_argument("--brand", default="OTHER")
    parser.add_argument("--city", default="OTHER")
    parser.add_argument("--version", help="model version (defaults to latest)")
    args = parser.parse_args(argv)

    from laptop_price.serving import predict_one

    if args.json:
        raw = sys.stdin.read() if args.json == "-" else args.json
        listing: dict[str, Any] = json.loads(raw)
    else:
        listing = {
            "RAM_SIZE": args.ram,
            "SSD_SIZE": args.ssd,
            "HDD_SIZE": args.hdd,
            "cpu_mark": args.cpu_mark,
            "gpu_g3d_mark": args.gpu_mark,
            "brand": args.brand,
            "city_grouped": args.city,
        }

    result = predict_one(listing, version=args.version)
    print(json.dumps(result, indent=2))
    return 0


def deals_main(argv: list[str] | None = None) -> int:
    """Rank listings where the model thinks the asking price is too low."""
    parser = argparse.ArgumentParser(prog="laptop-deals", description=deals_main.__doc__)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--version", help="model version (defaults to latest)")
    args = parser.parse_args(argv)

    from laptop_price.anomaly.deals import rank_deals
    from laptop_price.data import load_model_ready
    from laptop_price.models.registry import load_bundle

    matrix = load_model_ready()
    bundle = load_bundle(args.version)
    ranked = rank_deals(matrix, bundle, top_k=args.top)

    _print_header(f"Top {args.top} candidate bargains")
    columns = ["price_corrected", "predicted", "discount_pct", "brand", "city_grouped", "verdict"]
    with pd.option_context("display.width", 140):
        print(ranked[[c for c in columns if c in ranked.columns]].to_string(index=False))
    return 0


COMMANDS = {
    "pipeline": pipeline_main,
    "train": train_main,
    "predict": predict_main,
    "deals": deals_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python -m laptop_price {{{'|'.join(COMMANDS)}}} [options]")
        return 1
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
