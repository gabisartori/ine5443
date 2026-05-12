#!/usr/bin/env python3
"""
Avalia a accuracy de um CSV de previsoes binarias contra o GroundTruth.

Uso:
  python scripts/evaluate_test_accuracy.py \
    --ground-truth GroundTruth.csv \
    --predictions test.csv

Por padrao, a classe positiva no GroundTruth e MEL. Use --positive-classes
para ajustar o criterio, por exemplo:

  --positive-classes MEL,BCC,AKIEC
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize_image_id(value: str) -> str:
    return Path(value.strip()).stem


def parse_binary(value: str, threshold: float = 0.5) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n", ""}:
        return 0
    try:
        return 1 if float(text) >= threshold else 0
    except ValueError as exc:
        raise ValueError(f"Nao foi possivel interpretar valor binario: {value!r}") from exc


def load_ground_truth(path: Path, positive_classes: list[str]) -> dict[str, int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV vazio ou sem cabecalho: {path}")

        missing = [column for column in ["image", *positive_classes] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Colunas ausentes em {path}: {', '.join(missing)}")

        labels: dict[str, int] = {}
        for row in reader:
            image_id = normalize_image_id(row["image"])
            labels[image_id] = 1 if any(parse_binary(row[column]) for column in positive_classes) else 0
        return labels


def load_predictions(path: Path, threshold: float) -> dict[str, int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV vazio ou sem cabecalho: {path}")

        if "image" not in reader.fieldnames:
            raise ValueError(f"Coluna 'image' ausente em {path}")

        prediction_column = None
        for candidate in ("suspicious", "score"):
            if candidate in reader.fieldnames:
                prediction_column = candidate
                break

        if prediction_column is None:
            raise ValueError(f"Nenhuma coluna de previsao encontrada em {path} (esperado 'suspicious' ou 'score')")

        predictions: dict[str, int] = {}
        for row in reader:
            image_id = normalize_image_id(row["image"])
            predictions[image_id] = parse_binary(row[prediction_column], threshold=threshold)
        return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Calcula accuracy entre GroundTruth e previsoes binarias.")
    parser.add_argument("--ground-truth", default="GroundTruth.csv", help="CSV com as classes reais")
    parser.add_argument("--predictions", default="test.csv", help="CSV com as previsoes")
    parser.add_argument(
        "--positive-classes",
        default="MEL, BCC",
        help="Classes do GroundTruth que contam como positivas, separadas por virgula",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limite usado quando a coluna de previsao for numerica",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    predictions_path = Path(args.predictions)
    positive_classes = [column.strip() for column in args.positive_classes.split(",") if column.strip()]

    ground_truth = load_ground_truth(ground_truth_path, positive_classes)
    predictions = load_predictions(predictions_path, args.threshold)

    common_ids = sorted(set(ground_truth) & set(predictions))
    missing_in_predictions = sorted(set(ground_truth) - set(predictions))
    missing_in_ground_truth = sorted(set(predictions) - set(ground_truth))

    if not common_ids:
        raise ValueError("Nenhum image_id em comum entre GroundTruth e predictions.")

    correct = sum(1 for image_id in common_ids if ground_truth[image_id] == predictions[image_id])
    total = len(common_ids)
    accuracy = correct / total

    true_positive = sum(1 for image_id in common_ids if ground_truth[image_id] == 1 and predictions[image_id] == 1)
    true_negative = sum(1 for image_id in common_ids if ground_truth[image_id] == 0 and predictions[image_id] == 0)
    false_positive = sum(1 for image_id in common_ids if ground_truth[image_id] == 0 and predictions[image_id] == 1)
    false_negative = sum(1 for image_id in common_ids if ground_truth[image_id] == 1 and predictions[image_id] == 0)

    print(f"Total avaliados: {total}")
    print(f"Acertos: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {true_positive}  TN: {true_negative}  FP: {false_positive}  FN: {false_negative}")

    if missing_in_predictions:
        print(f"Sem previsao para {len(missing_in_predictions)} imagens")
    if missing_in_ground_truth:
        print(f"Sem GroundTruth para {len(missing_in_ground_truth)} imagens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())