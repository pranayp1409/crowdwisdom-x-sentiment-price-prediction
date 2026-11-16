from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentClassifier:
    vectorizer: HashingVectorizer
    classifier: SGDClassifier

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty(0, dtype=float)
        probabilities = self.classifier.predict_proba(
            self.vectorizer.transform(texts)
        )
        classes = list(self.classifier.classes_)
        try:
            negative_index = classes.index(0)
            positive_index = classes.index(1)
        except ValueError as error:
            raise ValueError(
                "The sentiment classifier must be trained with negative and positive labels."
            ) from error
        return probabilities[:, positive_index] - probabilities[:, negative_index]


def load_sentiment_classifier(path: Path) -> SentimentClassifier:
    model = joblib.load(path)
    if not isinstance(model, SentimentClassifier):
        raise ValueError(f"Sentiment model at {path} has an unexpected format.")
    return model


def train_sentiment_classifier(
    dataset_path: Path,
    model_path: Path,
    evaluation_path: Path,
    batch_size: int = 8_192,
) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Sentiment dataset does not exist: {dataset_path}")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2.")

    vectorizer = HashingVectorizer(
        n_features=2**20,
        ngram_range=(1, 2),
        alternate_sign=False,
        strip_accents="unicode",
    )
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        random_state=42,
        learning_rate="optimal",
    )
    train_buffers = {0: deque(), 1: deque()}
    train_rows = Counter()
    validation_rows = Counter()
    skipped_empty_rows = 0
    invalid_rows = 0
    first_batch = True
    half_batch = batch_size // 2

    def fit_batch(texts: list[str], labels: list[int]) -> None:
        nonlocal first_batch
        if not texts:
            return
        order = np.random.default_rng(42 + sum(train_rows.values())).permutation(len(texts))
        shuffled_texts = [texts[int(index)] for index in order]
        shuffled_labels = np.asarray([labels[int(index)] for index in order], dtype=np.int8)
        vectors = vectorizer.transform(shuffled_texts)
        if first_batch:
            classifier.partial_fit(vectors, shuffled_labels, classes=np.asarray([0, 1]))
            first_batch = False
        else:
            classifier.partial_fit(vectors, shuffled_labels)

    with dataset_path.open("r", encoding="latin-1", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 6:
                invalid_rows += 1
                continue
            try:
                raw_label = int(row[0])
            except ValueError:
                invalid_rows += 1
                continue
            if raw_label not in {0, 4}:
                invalid_rows += 1
                continue
            text = row[5].strip()
            if not text:
                skipped_empty_rows += 1
                continue
            label = 1 if raw_label == 4 else 0
            if _is_validation_text(text):
                validation_rows[label] += 1
                continue

            train_buffers[label].append(text)
            train_rows[label] += 1
            if min(len(train_buffers[0]), len(train_buffers[1])) >= half_batch:
                texts = [
                    train_buffers[0].popleft()
                    for _ in range(half_batch)
                ] + [
                    train_buffers[1].popleft()
                    for _ in range(half_batch)
                ]
                labels = [0] * half_batch + [1] * half_batch
                fit_batch(texts, labels)

    leftovers = [
        (text, label)
        for label in (0, 1)
        for text in train_buffers[label]
    ]
    if leftovers:
        fit_batch(
            [text for text, _ in leftovers],
            [label for _, label in leftovers],
        )
    if first_batch or not validation_rows[0] or not validation_rows[1]:
        raise ValueError(
            "The dataset must contain both sentiment classes in the training and "
            "held-out partitions."
        )

    model = SentimentClassifier(vectorizer=vectorizer, classifier=classifier)
    metrics = _evaluate_sentiment140(
        dataset_path,
        model,
        expected_validation_rows=sum(validation_rows.values()),
    )
    metrics.update(
        {
            "dataset_file": dataset_path.name,
            "train_rows": int(sum(train_rows.values())),
            "validation_rows": int(sum(validation_rows.values())),
            "skipped_empty_rows": int(skipped_empty_rows),
            "invalid_rows": int(invalid_rows),
            "split": "20% held out by stable hash of tweet text",
        }
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    evaluation_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _is_validation_text(text: str) -> bool:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % 5 == 0


def _evaluate_sentiment140(
    dataset_path: Path,
    model: SentimentClassifier,
    expected_validation_rows: int,
    batch_size: int = 8_192,
) -> dict[str, float | int]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    vader_confusion = np.zeros((2, 2), dtype=np.int64)
    vader_covered = 0
    vader_analyzer = SentimentIntensityAnalyzer()
    pending_texts: list[str] = []
    pending_labels: list[int] = []

    def evaluate_batch() -> None:
        nonlocal vader_covered
        if not pending_texts:
            return
        predictions = (model.score(pending_texts) >= 0).astype(np.int8)
        np.add.at(confusion, (pending_labels, predictions), 1)
        for text, label in zip(pending_texts, pending_labels):
            vader_score = vader_analyzer.polarity_scores(text)["compound"]
            if abs(vader_score) > 0.05:
                vader_label = int(vader_score > 0.05)
                vader_confusion[label, vader_label] += 1
                vader_covered += 1
        pending_texts.clear()
        pending_labels.clear()

    with dataset_path.open("r", encoding="latin-1", newline="") as stream:
        reader = csv.reader(stream)
        for row in reader:
            if len(row) != 6 or not row[5].strip():
                continue
            text = row[5].strip()
            if not _is_validation_text(text):
                continue
            raw_label = int(row[0])
            if raw_label not in {0, 4}:
                continue
            pending_texts.append(text)
            pending_labels.append(1 if raw_label == 4 else 0)
            if len(pending_texts) >= batch_size:
                evaluate_batch()
        evaluate_batch()

    evaluated_rows = int(confusion.sum())
    if evaluated_rows != expected_validation_rows:
        raise RuntimeError(
            "Held-out sentiment row count changed between training and evaluation."
        )
    return {
        "validation_accuracy": float(np.trace(confusion) / evaluated_rows),
        "validation_macro_f1": _macro_f1(confusion),
        "vader_covered_rows": int(vader_covered),
        "vader_coverage_pct": (
            float(vader_covered / evaluated_rows * 100) if evaluated_rows else 0.0
        ),
        "vader_accuracy_on_covered": (
            float(np.trace(vader_confusion) / vader_covered) if vader_covered else 0.0
        ),
        "validation_confusion_matrix": confusion.tolist(),
        "vader_confusion_matrix_on_covered": vader_confusion.tolist(),
    }


def _macro_f1(confusion: np.ndarray) -> float:
    scores = []
    for label in range(2):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum() - true_positive)
        false_negative = int(confusion[label, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))
