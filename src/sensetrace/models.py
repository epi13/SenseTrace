"""Small CPU-first model ladder for Phase 0."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .errors import SenseTraceError
from .hashing import sha256_bytes, sha256_json
from .metrics import evaluate_predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class FittedModel:
    name: str
    seed: int
    parameter_count: int
    model_hash: str
    predict: Any


def _torch_model(
    kind: str,
    trace_length: int,
    dense_width: int = 32,
    channels: list[int] | None = None,
    input_dim: int = 69,
):
    import torch.nn as nn

    widths = channels or [16, 32, 32, 64]
    if kind == "tiny_mlp":
        return nn.Sequential(
            nn.Linear(input_dim, dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, max(8, dense_width // 2)),
            nn.ReLU(),
            nn.Linear(max(8, dense_width // 2), 1),
        )
    layers: list[nn.Module] = []
    input_channels = 1
    kernels = [11, 7, 5, 3]
    for output_channels, kernel in zip(widths, kernels, strict=True):
        layers.extend([nn.Conv1d(input_channels, output_channels, kernel), nn.ReLU()])
        if output_channels != widths[-1]:
            layers.append(nn.MaxPool1d(2))
        input_channels = output_channels
    layers.extend(
        [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(widths[-1], dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, 1),
        ]
    )
    return nn.Sequential(*layers)


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def fit_model(
    name: str,
    traces: np.ndarray,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 256,
) -> FittedModel:
    set_seed(seed)
    train = partitions["train"]
    validation = partitions["validation"]
    if name == "logistic_regression":
        scaler = StandardScaler().fit(feature_matrix[train])
        model = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="lbfgs")
        model.fit(scaler.transform(feature_matrix[train]), labels[train])
        parameter_count = int(model.coef_.size + model.intercept_.size)

        def predict_logistic(values: np.ndarray) -> np.ndarray:
            return model.predict_proba(scaler.transform(values))[:, 1]

        model_hash = sha256_json(
            {
                "name": name,
                "seed": seed,
                "coef": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "scale_mean": scaler.mean_.tolist(),
                "scale_scale": scaler.scale_.tolist(),
            }
        )
        return FittedModel(name, seed, parameter_count, model_hash, predict_logistic)
    if name == "boosted_trees":
        model = HistGradientBoostingClassifier(
            max_iter=min(epochs, 80),
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(feature_matrix[train], labels[train])

        def predict_tree(values: np.ndarray) -> np.ndarray:
            return model.predict_proba(values)[:, 1]

        tree_count = sum(tree.get_n_leaf_nodes() for stage in model._predictors for tree in stage)
        model_hash = sha256_bytes(
            model.predict_proba(feature_matrix[train]).astype(np.float32).tobytes()
        )
        return FittedModel(name, seed, int(tree_count), model_hash, predict_tree)
    if not torch_available():
        raise SenseTraceError(
            "PyTorch is required for tiny_mlp and tiny_cnn; install sensetrace[ml]"
        )
    import torch
    import torch.nn as nn

    if name == "tiny_mlp":
        model = _torch_model("tiny_mlp", traces.shape[1], input_dim=feature_matrix.shape[1])
        x_values = feature_matrix
        input_transform = StandardScaler().fit(x_values[train])
        x_values = input_transform.transform(x_values).astype(np.float32)
        x_train = torch.from_numpy(x_values[train])
        x_validation = torch.from_numpy(x_values[validation])
    elif name == "tiny_cnn":
        model = _torch_model("tiny_cnn", traces.shape[1])
        input_transform = StandardScaler().fit(traces[train])
        x_values = input_transform.transform(traces).astype(np.float32)[:, None, :]
        x_train = torch.from_numpy(x_values[train])
        x_validation = torch.from_numpy(x_values[validation])
    else:
        raise SenseTraceError(f"unknown model {name}")
    y_train = torch.from_numpy(labels[train].astype(np.float32)[:, None])
    y_validation = torch.from_numpy(labels[validation].astype(np.float32)[:, None])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_loss = float("inf")
    stale = 0
    torch.set_num_threads(1)
    for _ in range(min(epochs, 50)):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_validation), y_validation))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(values: np.ndarray) -> np.ndarray:
        transformed = input_transform.transform(values).astype(np.float32)
        if name == "tiny_cnn":
            transformed = transformed[:, None, :]
        with torch.no_grad():
            logits = model(torch.from_numpy(transformed)).reshape(-1)
            return torch.sigmoid(logits).cpu().numpy()

    model_hash = sha256_bytes(
        b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
    )
    return FittedModel(
        name,
        seed,
        _parameter_count(model),
        model_hash,
        predict,
    )


def train_and_evaluate(
    name: str,
    traces: np.ndarray,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    *,
    seeds: list[int],
    dataset_fingerprint: str,
    split_fingerprint: str,
    epochs: int,
    patience: int,
    batch_size: int,
    groups: np.ndarray | None = None,
    ci_unit: str = "sample",
    bootstrap_repetitions: int = 400,
) -> dict[str, Any]:
    results = []
    for seed in seeds:
        fitted = fit_model(
            name,
            traces,
            feature_matrix,
            labels,
            partitions,
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
        )
        test_indices = partitions["test"]
        results.append(
            evaluate_predictions(
                labels[test_indices],
                fitted.predict(
                    feature_matrix[test_indices]
                    if name in {"logistic_regression", "boosted_trees", "tiny_mlp"}
                    else traces[test_indices]
                ),
                seed=seed,
                parameter_count=fitted.parameter_count,
                model_name=name,
                dataset_fingerprint=dataset_fingerprint,
                split_fingerprint=split_fingerprint,
                groups=None if groups is None else groups[test_indices],
                ci_unit=ci_unit,
                bootstrap_repetitions=bootstrap_repetitions,
            )
        )
        results[-1]["model_hash"] = fitted.model_hash
    scores = [item["balanced_accuracy"] for item in results]
    aucs = [item["auroc"] for item in results]
    return {
        "model": name,
        "parameter_count": results[0]["parameter_count"],
        "seeds": seeds,
        "runs": results,
        "summary": {
            "balanced_accuracy_mean": float(np.mean(scores)),
            "balanced_accuracy_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "auroc_mean": float(np.mean(aucs)),
            "auroc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        },
    }
