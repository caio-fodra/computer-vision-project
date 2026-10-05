from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, MNIST_DATA_PATH, MODEL_PATH
from hand_features import FEATURE_SIZE

MIN_SAMPLES_PER_CLASS = 50
RANDOM_STATE = 42


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != FEATURE_SIZE + 1:
                print(f"Linha {line_number} ignorada: {len(row)} colunas "
                      f"(esperado {FEATURE_SIZE + 1})")
                continue
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])

    return np.array(features, dtype=np.float32), np.array(labels)


def resolve_paths(requested: list[Path] | None) -> list[Path]:
    if requested:
        missing = [str(p) for p in requested if not p.exists()]
        if missing:
            print("Arquivos nao encontrados: " + ", ".join(missing))
            return []
        return requested

    paths = [p for p in (DATA_PATH, MNIST_DATA_PATH) if p.exists()]
    if not paths:
        print("Nenhum CSV de landmarks encontrado. Rode antes collect_data.py")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o classificador de landmarks.")
    parser.add_argument("--data", nargs="+", type=Path, default=None,
                        help="CSVs de landmarks (padrao: todos os que existirem)")
    paths = resolve_paths(parser.parse_args().data)
    if not paths:
        return 1

    parts_X, parts_y = [], []
    for path in paths:
        X_part, y_part = load_dataset(path)
        print(f"{path}: {len(y_part)} exemplos")
        if len(y_part):
            parts_X.append(X_part)
            parts_y.append(y_part)

    if not parts_y:
        print("Os CSVs estao vazios.")
        return 1

    X = np.concatenate(parts_X)
    y = np.concatenate(parts_y)
    counts = Counter(y.tolist())

    print("\nExemplos por letra:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    too_few = [label for label, c in counts.items() if c < MIN_SAMPLES_PER_CLASS]
    if too_few:
        print(f"\nIgnorando letras com menos de {MIN_SAMPLES_PER_CLASS} exemplos: "
              f"{', '.join(sorted(too_few))}")
        keep = ~np.isin(y, too_few)
        X, y = X[keep], y[keep]

    if len(set(y.tolist())) < 2:
        print("\nE preciso ter pelo menos 2 letras com exemplos suficientes.")
        return 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                      max_iter=1000, random_state=RANDOM_STATE),
    )

    print(f"\nTreinando com {len(X_train)} exemplos, testando com {len(X_test)}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nRelatorio no conjunto de teste:")
    print(classification_report(y_test, predictions, zero_division=0))

    classes = list(model.classes_)
    print("Matriz de confusao (linhas = real, colunas = previsto):")
    print("    " + " ".join(f"{c:>4}" for c in classes))
    for label, row in zip(classes, confusion_matrix(y_test, predictions, labels=classes)):
        print(f"{label:>4}" + " ".join(f"{v:>4}" for v in row))

    print("\nAtencao: frames gravados em sequencia sao muito parecidos entre si,")
    print("entao essa acuracia tende a ser otimista. O teste real e na webcam.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_size": FEATURE_SIZE, "labels": classes}, MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())