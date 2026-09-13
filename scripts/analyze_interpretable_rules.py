#!/usr/bin/env python3
"""Quantifies the simplicity-vs-performance trade-off of using a plain
decision tree (fully human-readable rules) instead of the ensemble this
project ships, and prints the shallowest tree that still says something
useful. Backs the analysis in docs/10_analise_resultados.md, section 10.2.

Usage: python scripts/analyze_interpretable_rules.py
Requires a cached feature table (`python -m pdm.cli features` first).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from pdm.config import load_config
from pdm.data.io import load_feature_table
from pdm.models.train import feature_columns

READABLE_DEPTH = 4
DEPTHS_TO_COMPARE = [2, 3, 4, 5, 6, 8, None]


def main() -> None:
    cfg = load_config()
    table = load_feature_table(cfg.paths.processed_dir)
    cols = feature_columns(table)
    X, y = table[cols], table["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=cfg.random_seed
    )

    print("=== Trade-off: profundidade da árvore vs. desempenho ===")
    for depth in DEPTHS_TO_COMPARE:
        tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=cfg.random_seed)
        cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        tree.fit(X_train, y_train)
        test_f1 = f1_score(y_test, tree.predict(X_test), average="macro")
        label = depth if depth else "sem limite"
        print(
            f"profundidade={label!s:>10}  CV F1-macro={cv_scores.mean():.4f}  "
            f"F1 teste={test_f1:.4f}  folhas={tree.get_n_leaves():4d}  nós={tree.tree_.node_count:4d}"
        )

    print(f"\n=== Árvore rasa (profundidade {READABLE_DEPTH}): regras completas ===")
    tree = DecisionTreeClassifier(
        max_depth=READABLE_DEPTH, class_weight="balanced", random_state=cfg.random_seed
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print(f"F1-macro em teste: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred))
    print(export_text(tree, feature_names=list(X.columns), max_depth=READABLE_DEPTH))

    print("=== Importância de feature (Gini) na árvore rasa ===")
    importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(15).to_string())


if __name__ == "__main__":
    main()
