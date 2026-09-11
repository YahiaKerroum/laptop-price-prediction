"""Spec-inconsistency detection driven by the mined association rules.

If ``{RTX 4060} -> {16GB RAM}`` holds at 82% confidence across the market, a
listing claiming *RTX 4060 + 4GB DDR3* is probably mistyped or fraudulent. That
turns the association-rules deliverable - which otherwise sits on its own - into
an input to anomaly detection, and it is the piece that makes the three models
read as one project.

Rules are read from ``reports/rules/*.csv``, which ``notebooks/08_association_rules.ipynb``
writes, so this module has no opinion about how they were mined.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from laptop_price import paths
from laptop_price.config import CONFIG


def _parse_itemset(value: object) -> frozenset[str]:
    """Parse mlxtend's ``frozenset({'a', 'b'})`` repr back into a set."""
    if isinstance(value, frozenset):
        return value
    text = str(value).strip()
    if text.startswith("frozenset("):
        text = text[len("frozenset(") : -1]
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return frozenset()
    if isinstance(parsed, set | frozenset | list | tuple):
        return frozenset(str(item) for item in parsed)
    return frozenset({str(parsed)})


def load_rules(
    directory: Path | None = None,
    *,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    """Load every mined rule file and keep the high-confidence ones.

    Returns an empty frame (not an error) when the notebook has not been run, so
    a consistency check can degrade gracefully rather than break the pipeline.
    """
    min_confidence = (
        CONFIG.association.inconsistency_confidence if min_confidence is None else min_confidence
    )
    root = directory or paths.RULES_DIR
    if not root.exists():
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    frames = []
    for path in sorted(root.glob("rules_*.csv")):
        frame = pd.read_csv(path)
        frame["source"] = path.stem
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = pd.concat(frames, ignore_index=True)
    rules["antecedents"] = rules["antecedents"].map(_parse_itemset)
    rules["consequents"] = rules["consequents"].map(_parse_itemset)
    return rules[rules["confidence"] >= min_confidence].reset_index(drop=True)


def check_spec_consistency(
    transactions: pd.DataFrame,
    rules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Flag listings that violate a high-confidence rule.

    Parameters
    ----------
    transactions
        One-hot encoded listing features, in the same item vocabulary the rules
        were mined over (the ``build_experiment_df`` output of notebook 08).

    Returns
    -------
    DataFrame
        ``n_rules_violated``, the worst confidence violated, and a readable
        ``violations`` description per listing.
    """
    rules = load_rules() if rules is None else rules
    result = pd.DataFrame(
        {
            "n_rules_violated": 0,
            "max_violated_confidence": 0.0,
            "violations": [[] for _ in range(len(transactions))],
        },
        index=transactions.index,
    )
    if rules.empty or transactions.empty:
        return result

    vocabulary = set(transactions.columns)
    for rule in rules.itertuples(index=False):
        antecedents = set(rule.antecedents) & vocabulary
        consequents = set(rule.consequents) & vocabulary
        if not antecedents or not consequents:
            continue

        # The antecedent holds ...
        holds = transactions[list(antecedents)].all(axis=1)
        # ... but the consequent does not.
        violated = holds & ~transactions[list(consequents)].all(axis=1)
        if not violated.any():
            continue

        label = f"{sorted(antecedents)} -> {sorted(consequents)} (conf={rule.confidence:.2f})"
        result.loc[violated, "n_rules_violated"] += 1
        result.loc[violated, "max_violated_confidence"] = result.loc[
            violated, "max_violated_confidence"
        ].clip(lower=float(rule.confidence))
        for index in result.index[violated]:
            result.at[index, "violations"].append(label)

    result["is_inconsistent"] = result["n_rules_violated"] > 0
    return result


def summarise_violations(consistency: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Most frequently violated rules, for the report."""
    exploded = consistency["violations"].explode().dropna()
    return exploded.value_counts().head(top_n)


__all__ = [
    "check_spec_consistency",
    "load_rules",
    "summarise_violations",
]
