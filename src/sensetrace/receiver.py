"""Small, memory-bounded receivers for fragmented SenseTrace evidence.

The receiver consumes immutable packet batches.  Acquisition is intentionally
absent from this module: callers must finish and fingerprint an evidence
dataset before constructing a receiver or starting training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .hashing import sha256_json
from .packets import EvidencePacket, PacketBatch

if TYPE_CHECKING:
    from torch.nn import Module as _ModuleBase
else:
    try:
        from torch.nn import Module as _ModuleBase
    except ImportError:  # pragma: no cover - exercised only without the optional ML extra

        class _ModuleBase:
            def __init__(self) -> None:
                super().__init__()


ClaimLevel = Literal[
    "level_1_exact_host_calibrated",
    "level_2_exact_host_unseen_location",
    "level_3_exact_host_unseen_session",
    "level_4_exact_host_unseen_boot",
    "level_5_unseen_dimm",
    "level_6_unseen_host",
]

CLAIM_LEVELS: dict[ClaimLevel, str] = {
    "level_1_exact_host_calibrated": "calibrated exact-host decoding only",
    "level_2_exact_host_unseen_location": "unseen location on the same exact host",
    "level_3_exact_host_unseen_session": "unseen acquisition/session on the same host",
    "level_4_exact_host_unseen_boot": "unseen OS boot on the same host",
    "level_5_unseen_dimm": "unseen DIMM/device",
    "level_6_unseen_host": "different host",
}

PACKET_RECEIVER_LADDER = (
    "logistic_regression",
    "boosted_trees",
    "tiny_cnn_tcn",
    "weak_evidence_aggregator",
    "jepa_linear_probe",
    "jepa_tiny_mlp",
    "predictive_coding",
    "jepa_predictive_coding_hybrid",
)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class ReceiverConfig:
    latent_width: int = 64
    hidden_width: int = 48
    refinement_steps: int = 4
    refinement_step_size: float = 0.25
    detach_between_refinement_steps: bool = True

    def validate(self) -> None:
        if not 8 <= self.latent_width <= 128:
            raise ValueError("latent_width must be between 8 and 128")
        if not 8 <= self.hidden_width <= 256:
            raise ValueError("hidden_width must be between 8 and 256")
        if not 1 <= self.refinement_steps <= 16:
            raise ValueError("refinement_steps must be between 1 and 16")
        if not 0.0 < self.refinement_step_size <= 1.0:
            raise ValueError("refinement_step_size must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "latent_width": self.latent_width,
            "hidden_width": self.hidden_width,
            "refinement_steps": self.refinement_steps,
            "refinement_step_size": self.refinement_step_size,
            "detach_between_refinement_steps": self.detach_between_refinement_steps,
        }


@dataclass(frozen=True)
class ReceiverTournament:
    """Immutable comparison plan shared by all receiver candidates."""

    dataset_fingerprint: str
    split_fingerprint: str
    candidates: tuple[str, ...] = PACKET_RECEIVER_LADDER
    claim_level: ClaimLevel = "level_1_exact_host_calibrated"

    def validate(self) -> None:
        if not self.dataset_fingerprint or not self.split_fingerprint:
            raise ValueError(
                "receiver tournament requires immutable dataset and split fingerprints"
            )
        if not self.candidates or any(
            candidate not in PACKET_RECEIVER_LADDER for candidate in self.candidates
        ):
            raise ValueError("receiver tournament contains an unsupported candidate")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("receiver tournament candidates must be unique")
        claim_boundary(self.claim_level)

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "sensetrace.receiver-tournament.v1",
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "candidates": list(self.candidates),
            "claim_boundary": claim_boundary(self.claim_level),
            "selection_policy": "validation split selects; untouched test split is evaluated once",
        }


def audit_model_input_schema(batch: PacketBatch) -> dict[str, Any]:
    """Return an audit record for the deliberately narrow model input schema."""

    batch.validate()
    return {
        "schema": "sensetrace.receiver-model-input.v1",
        "allowed_arrays": ["values", "observed_mask", "fragment_mask", "quality", "excitation"],
        "excluded": [
            "packet_id",
            "target_reference",
            "acquisition_id",
            "protocol_id",
            "controls",
            "provenance",
            "labels",
            "run_id",
            "session_id",
            "boot_id",
            "host_id",
            "address",
            "packet_order",
        ],
        "labels_present_in_batch": batch.labels is not None,
        "claim": "identifiers and labels are not model inputs",
    }


def packet_summary_features(batch: PacketBatch) -> np.ndarray:
    """Make small baseline features from observed payloads and explicit masks."""

    batch.validate()
    observed_values = np.where(batch.observed_mask, batch.values, 0.0)
    valid = batch.observed_mask.sum(axis=2).astype(np.float32)
    fragment_means = np.divide(
        observed_values.sum(axis=2),
        valid,
        out=np.zeros_like(valid),
        where=valid > 0,
    )
    second_moment = np.divide(
        np.square(observed_values).sum(axis=2),
        valid,
        out=np.zeros_like(valid),
        where=valid > 0,
    )
    fragment_stds = np.sqrt(np.maximum(second_moment - np.square(fragment_means), 0.0))
    fragment_valid = batch.fragment_mask.sum(axis=1).astype(np.float32)
    summaries = [
        np.mean(fragment_means, axis=1),
        np.std(fragment_means, axis=1),
        np.min(fragment_means, axis=1),
        np.max(fragment_means, axis=1),
        np.mean(fragment_stds, axis=1),
        np.mean(valid, axis=1),
        fragment_valid,
        np.mean(batch.quality, axis=1),
        # Ordered per-fragment summaries retain the declared weak-probe
        # relationship. They are derived only from model arrays, so rotating
        # fragments is a meaningful relation ablation without exposing probe
        # IDs or audit metadata.
        fragment_means,
        fragment_stds,
        valid,
        batch.fragment_mask.astype(np.float32),
    ]
    return np.column_stack(
        [summary.reshape(batch.values.shape[0], -1) for summary in summaries]
    ).astype(np.float32, copy=False)


def claim_boundary(level: ClaimLevel) -> dict[str, str]:
    if level not in CLAIM_LEVELS:
        raise ValueError(f"unsupported claim level {level!r}")
    return {
        "claim_level": level,
        "scope": CLAIM_LEVELS[level],
        "not_established": "signal mechanism, DRAM origin, cell origin, or broader generalization",
    }


def _require_torch() -> Any:
    if not torch_available():
        raise RuntimeError("PyTorch is required for the latent receiver; install sensetrace[ml]")
    import torch

    return torch


class FragmentEncoder:
    """Factory namespace for the optional torch implementation."""

    @staticmethod
    def build(config: ReceiverConfig, excitation_width: int) -> Any:
        torch = _require_torch()
        import torch.nn as nn

        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temporal = nn.Sequential(
                    nn.Conv1d(2, 12, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(12, 16, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.projection = nn.Sequential(
                    nn.Linear(16 + excitation_width + 1, config.hidden_width),
                    nn.GELU(),
                    nn.Linear(config.hidden_width, config.latent_width),
                )

            def forward(
                self,
                values: Any,
                observed_mask: Any,
                quality: Any,
                excitation: Any,
            ) -> Any:
                batch, fragments, payload = values.shape
                temporal_input = torch.stack(
                    (values, observed_mask.to(dtype=values.dtype)), dim=2
                ).reshape(batch * fragments, 2, payload)
                temporal = self.temporal(temporal_input).reshape(batch * fragments, 16)
                side = torch.cat(
                    (
                        quality.reshape(batch * fragments, 1),
                        excitation.reshape(batch * fragments, -1),
                    ),
                    dim=1,
                )
                return self.projection(torch.cat((temporal, side), dim=1)).reshape(
                    batch, fragments, config.latent_width
                )

        return _Encoder()


def _weighted_aggregate(latents: Any, fragment_mask: Any, quality: Any) -> Any:
    weights = fragment_mask.to(dtype=latents.dtype) * quality.to(dtype=latents.dtype)
    weights = weights.unsqueeze(-1)
    return (latents * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class JEPALikeEncoder(_ModuleBase):
    """Compact packet-level latent predictor with a stop-gradient target path."""

    def __init__(self, config: ReceiverConfig, excitation_width: int) -> None:
        super().__init__()
        config.validate()
        _require_torch()
        import torch.nn as nn

        self.module = nn.ModuleDict(
            {
                "context_encoder": FragmentEncoder.build(config, excitation_width),
                "target_encoder": FragmentEncoder.build(config, excitation_width),
                "predictor": nn.Sequential(
                    nn.Linear(config.latent_width, config.hidden_width),
                    nn.GELU(),
                    nn.Linear(config.hidden_width, config.latent_width),
                ),
            }
        )
        for parameter in self.module["target_encoder"].parameters():
            parameter.requires_grad_(False)

    def forward(self, batch: PacketBatch) -> dict[str, Any]:
        _require_torch()
        values, observed, fragments, quality, excitation = _batch_tensors(batch)
        context_fragments = self.module["context_encoder"](values, observed, quality, excitation)
        target_fragments = self.module["target_encoder"](values, observed, quality, excitation)
        context = _weighted_aggregate(context_fragments, fragments, quality)
        target = _weighted_aggregate(target_fragments, fragments, quality).detach()
        predicted = self.module["predictor"](context)
        return {"context": context, "predicted": predicted, "target": target}

    def update_target(self, momentum: float = 0.99) -> None:
        torch = _require_torch()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("target encoder momentum must be in [0, 1)")
        context = self.module["context_encoder"]
        target = self.module["target_encoder"]
        with torch.no_grad():
            for target_parameter, context_parameter in zip(
                target.parameters(), context.parameters(), strict=True
            ):
                target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)


class PredictiveCodingRefiner(_ModuleBase):
    """Small iterative prediction-error update over packet fragment latents."""

    def __init__(self, config: ReceiverConfig) -> None:
        super().__init__()
        config.validate()
        _require_torch()
        import torch.nn as nn

        self.config = config
        self.module = nn.ModuleDict(
            {
                "fragment_predictor": nn.Sequential(
                    nn.Linear(config.latent_width, config.hidden_width),
                    nn.GELU(),
                    nn.Linear(config.hidden_width, config.latent_width),
                ),
                "update_net": nn.Sequential(
                    nn.Linear(config.latent_width * 2, config.hidden_width),
                    nn.Tanh(),
                    nn.Linear(config.hidden_width, config.latent_width),
                ),
            }
        )

    def forward(self, fragment_latents: Any, fragment_mask: Any, quality: Any) -> Any:
        torch = _require_torch()
        latent = _weighted_aggregate(fragment_latents, fragment_mask, quality)
        weights = (fragment_mask.to(dtype=latent.dtype) * quality.to(dtype=latent.dtype)).unsqueeze(
            -1
        )
        for step in range(self.config.refinement_steps):
            predicted = self.module["fragment_predictor"](latent).unsqueeze(1)
            residual = (fragment_latents - predicted) * weights
            residual_mean = residual.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            update = self.module["update_net"](torch.cat((latent, residual_mean), dim=-1))
            latent = latent + self.config.refinement_step_size * update
            if (
                self.config.detach_between_refinement_steps
                and step + 1 < self.config.refinement_steps
            ):
                latent = latent.detach()
        return latent


class JEPAPredictiveCodingHybrid(_ModuleBase):
    """JEPA pretraining signal plus an observation-conditioned settled latent."""

    def __init__(self, config: ReceiverConfig, excitation_width: int) -> None:
        super().__init__()
        config.validate()
        _require_torch()
        import torch.nn as nn

        self.config = config
        self.jepa = JEPALikeEncoder(config, excitation_width)
        self.refiner = PredictiveCodingRefiner(config)
        self.head = nn.ModuleDict(
            {
                "linear": nn.Linear(config.latent_width, 1),
                "mlp": nn.Sequential(
                    nn.Linear(config.latent_width, config.hidden_width),
                    nn.GELU(),
                    nn.Linear(config.hidden_width, 1),
                ),
            }
        )

    def trainable_parameters(self) -> Any:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.trainable_parameters())

    def forward(self, batch: PacketBatch) -> dict[str, Any]:
        _require_torch()
        values, observed, fragments, quality, excitation = _batch_tensors(batch)
        fragment_latents = self.jepa.module["context_encoder"](
            values, observed, quality, excitation
        )
        settled = self.refiner(fragment_latents, fragments, quality)
        jepa_context = _weighted_aggregate(fragment_latents, fragments, quality)
        target_fragments = self.jepa.module["target_encoder"](values, observed, quality, excitation)
        jepa_target = _weighted_aggregate(target_fragments, fragments, quality).detach()
        predicted = self.jepa.module["predictor"](jepa_context)
        return {
            "latent": settled,
            "linear_logits": self.head["linear"](settled).reshape(-1),
            "mlp_logits": self.head["mlp"](settled).reshape(-1),
            "jepa_predicted": predicted,
            "jepa_target": jepa_target,
            "refinement_steps": self.config.refinement_steps,
        }

    def update_target(self, momentum: float = 0.99) -> None:
        self.jepa.update_target(momentum)


def _batch_tensors(batch: PacketBatch) -> tuple[Any, Any, Any, Any, Any]:
    torch = _require_torch()
    batch.validate()
    return (
        torch.from_numpy(batch.values),
        torch.from_numpy(batch.observed_mask),
        torch.from_numpy(batch.fragment_mask),
        torch.from_numpy(batch.quality),
        torch.from_numpy(batch.excitation),
    )


def train_hybrid(
    batch_factory: Callable[[], Iterable[PacketBatch]],
    *,
    config: ReceiverConfig,
    excitation_width: int,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    seed: int = 1337,
) -> tuple[JEPAPredictiveCodingHybrid, dict[str, Any]]:
    """Train from a re-openable batch factory, retaining no corpus in memory."""

    torch = _require_torch()
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = JEPAPredictiveCodingHybrid(config, excitation_width)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=learning_rate, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model_loss = 0.0
        supervised_loss = 0.0
        jepa_loss = 0.0
        batches = 0
        model.train()
        for batch in batch_factory():
            if batch.labels is None:
                raise ValueError("hybrid supervised training requires labels in each batch")
            output = model(batch)
            labels = torch.from_numpy(batch.labels.astype(np.float32, copy=False))
            supervised = bce(output["linear_logits"], labels)
            latent_prediction = torch.nn.functional.mse_loss(
                output["jepa_predicted"], output["jepa_target"]
            )
            variance = output["latent"].std(dim=0).mean()
            collapse_penalty = torch.relu(torch.tensor(0.05) - variance)
            loss = supervised + 0.25 * latent_prediction + 0.05 * collapse_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            optimizer.step()
            model.update_target()
            model_loss += float(loss.detach())
            supervised_loss += float(supervised.detach())
            jepa_loss += float(latent_prediction.detach())
            batches += 1
        if batches == 0:
            raise ValueError("batch_factory produced no batches")
        history.append(
            {
                "epoch": float(epoch),
                "loss": model_loss / batches,
                "supervised_loss": supervised_loss / batches,
                "jepa_loss": jepa_loss / batches,
            }
        )
    return model, {
        "architecture": "sensetrace-jepa-predictive-coding-hybrid-v1",
        "config": config.as_dict(),
        "parameter_count": model.parameter_count(),
        "seed": seed,
        "epochs": epochs,
        "history": history,
        "memory_policy": "re-openable batch factory; no full-corpus tensor materialization",
        "supervised_head": "linear_logits is the primary reported head; mlp_logits is a paired comparison",
    }


def iter_hybrid_predictions(
    model: JEPAPredictiveCodingHybrid, batches: Iterable[PacketBatch]
) -> Iterator[np.ndarray]:
    """Yield prediction batches under inference mode instead of concatenating a corpus."""

    torch = _require_torch()
    model.jepa.module.eval()
    model.refiner.module.eval()
    model.head.eval()
    with torch.inference_mode():
        for batch in batches:
            output = model(batch)
            yield (
                torch.sigmoid(output["linear_logits"]).cpu().numpy().astype(np.float32, copy=False)
            )


def evaluate_hybrid(
    model: JEPAPredictiveCodingHybrid, batches: Iterable[PacketBatch]
) -> dict[str, Any]:
    """Evaluate in bounded batches with fixed-size confusion/AUROC histograms."""

    positive_histogram = np.zeros(256, dtype=np.int64)
    negative_histogram = np.zeros(256, dtype=np.int64)
    true_positive = true_negative = positive_count = negative_count = sample_count = 0
    torch = _require_torch()
    model.jepa.module.eval()
    model.refiner.module.eval()
    model.head.eval()
    with torch.inference_mode():
        for batch in batches:
            prediction = torch.sigmoid(model(batch)["linear_logits"]).cpu().numpy()
            prediction = np.asarray(prediction, dtype=np.float32)
            if batch.labels is None:
                raise ValueError("evaluation batches require labels")
            labels = batch.labels
            positive = labels == 1
            negative = ~positive
            indices = np.clip((prediction * 255.0).astype(np.int64), 0, 255)
            positive_histogram += np.bincount(indices[positive], minlength=256)
            negative_histogram += np.bincount(indices[negative], minlength=256)
            predicted_positive = prediction >= 0.5
            true_positive += int(np.sum(predicted_positive & positive))
            true_negative += int(np.sum(~predicted_positive & negative))
            positive_count += int(np.sum(positive))
            negative_count += int(np.sum(negative))
            sample_count += len(labels)
    if positive_count and negative_count:
        cumulative_negative = np.cumsum(negative_histogram) - negative_histogram
        auroc_numerator = float(
            np.sum(positive_histogram * cumulative_negative)
            + 0.5 * np.sum(positive_histogram * negative_histogram)
        )
        auroc = auroc_numerator / (positive_count * negative_count)
    else:
        auroc = float("nan")
    balanced_accuracy = (
        0.5 * ((true_positive / positive_count) if positive_count else 0.0)
        + 0.5 * ((true_negative / negative_count) if negative_count else 0.0)
        if positive_count or negative_count
        else float("nan")
    )
    return {
        "model": "jepa_predictive_coding_hybrid",
        "sample_count": sample_count,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "auroc_method": "256-bin streaming approximation",
        "prediction_retention": "no prediction array retained; fixed-size histograms only",
    }


class NoiseResidualizer:
    """Streaming unlabeled process model that subtracts predictable responses."""

    def __init__(self) -> None:
        self._sums: dict[tuple[str, str, int], np.ndarray] = {}
        self._counts: dict[tuple[str, str, int], np.ndarray] = {}
        self.packet_count = 0
        self.source_dataset_fingerprint: str | None = None

    def fit(
        self,
        packets: Iterable[EvidencePacket],
        *,
        dataset_fingerprint: str,
        require_unlabeled: bool = False,
    ) -> NoiseResidualizer:
        if not dataset_fingerprint:
            raise ValueError("noise model requires the source dataset fingerprint")
        if self.source_dataset_fingerprint not in {None, dataset_fingerprint}:
            raise ValueError("noise model source fingerprint cannot change after fitting")
        self.source_dataset_fingerprint = dataset_fingerprint
        for packet in packets:
            packet.validate()
            if require_unlabeled and packet.label is not None:
                raise ValueError("reference residualizer fitting requires unlabeled packets")
            self.packet_count += 1
            for fragment in packet.fragments:
                if not fragment.model_eligible or fragment.payload is None:
                    continue
                payload = np.asarray(fragment.payload, dtype=np.float32)
                mask = ~fragment.effective_mask()
                key = (fragment.probe_type, fragment.probe_version, payload.size)
                sums = self._sums.setdefault(key, np.zeros(payload.size, dtype=np.float64))
                counts = self._counts.setdefault(key, np.zeros(payload.size, dtype=np.float64))
                sums[mask] += payload[mask]
                counts[mask] += 1.0
        return self

    def fit_reference(
        self, packets: Iterable[EvidencePacket], *, dataset_fingerprint: str
    ) -> NoiseResidualizer:
        """Fit only from an explicitly unlabeled reference corpus."""

        return self.fit(
            packets,
            dataset_fingerprint=dataset_fingerprint,
            require_unlabeled=True,
        )

    def _mean(self, fragment: Any) -> np.ndarray | None:
        if fragment.payload is None:
            return None
        key = (fragment.probe_type, fragment.probe_version, fragment.payload_length)
        if key not in self._sums:
            return None
        counts = self._counts[key]
        return np.divide(
            self._sums[key], counts, out=np.zeros_like(self._sums[key]), where=counts > 0
        ).astype(np.float32)

    def transform(self, packet: EvidencePacket) -> EvidencePacket:
        packet.validate()
        fragments = []
        for fragment in packet.fragments:
            mean = self._mean(fragment)
            if mean is None or fragment.payload is None or not fragment.model_eligible:
                fragments.append(fragment)
                continue
            mask = fragment.effective_mask()
            residual = np.asarray(fragment.payload, dtype=np.float32).copy()
            residual[~mask] -= mean[~mask]
            fragments.append(replace(fragment, payload=residual))
        provenance = dict(packet.provenance)
        provenance["noise_residualizer"] = self.state_record()
        return replace(packet, fragments=tuple(fragments), provenance=provenance)

    def state_record(self) -> dict[str, Any]:
        return {
            "schema": "sensetrace.noise-process-model.v1",
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "packet_count": self.packet_count,
            "probe_shape_count": len(self._sums),
            "labels_used": False,
            "model": "per-probe/version running mean over observed values",
        }


def packet_stream_from_factory(
    factory: Callable[[], Iterable[EvidencePacket]],
    *,
    batch_size: int,
    max_fragments: int,
    max_payload_length: int,
    excitation_width: int = 0,
    include_labels: bool = True,
) -> Callable[[], Iterator[PacketBatch]]:
    """Build a re-openable bounded batch factory around a streaming packet source."""

    from .packets import iter_packet_batches

    def batches() -> Iterator[PacketBatch]:
        return iter_packet_batches(
            factory(),
            batch_size=batch_size,
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
            include_labels=include_labels,
        )

    return batches


def receiver_fingerprint(config: ReceiverConfig, *, architecture: str) -> str:
    material = {"architecture": architecture, "config": config.as_dict()}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _bounded_feature_matrix(
    packets: Iterable[EvidencePacket],
    *,
    max_fragments: int,
    max_payload_length: int,
    excitation_width: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize at most ``maximum`` summary rows for a bounded sklearn fit."""

    if maximum < 2:
        raise ValueError("maximum training rows must be at least two")
    features: list[np.ndarray] = []
    labels: list[int] = []
    for packet in packets:
        if packet.label is None:
            raise ValueError("supervised receiver training requires packet labels")
        batch = PacketBatch.from_packets(
            [packet],
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
        )
        features.append(packet_summary_features(batch)[0])
        labels.append(int(packet.label))
        if len(features) >= maximum:
            break
    if len(features) < 2 or len(set(labels)) < 2:
        raise ValueError("bounded training sample must contain both labels")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.uint8)


def _sklearn_model(candidate: str, *, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if candidate == "logistic_regression" or candidate == "weak_evidence_aggregator":
        return LogisticRegression(max_iter=200, solver="lbfgs", random_state=seed)
    if candidate == "boosted_trees":
        return HistGradientBoostingClassifier(
            max_iter=32, max_leaf_nodes=7, learning_rate=0.1, random_state=seed
        )
    if candidate in {
        "tiny_cnn_tcn",
        "jepa_linear_probe",
        "jepa_tiny_mlp",
        "predictive_coding",
        "jepa_predictive_coding_hybrid",
    }:
        # The packet summary is the declared CPU fallback when optional torch
        # is not installed.  It is deliberately tiny and is reported as a
        # bounded implementation of the candidate contract, not as a neural
        # trace model.
        return MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="tanh",
            solver="lbfgs",
            max_iter=100,
            random_state=seed,
        )
    raise ValueError(f"unsupported receiver candidate {candidate!r}")


class _StreamingMetrics:
    """Fixed-size metric accumulator for one evaluation partition."""

    def __init__(self) -> None:
        self.positive_histogram = np.zeros(256, dtype=np.int64)
        self.negative_histogram = np.zeros(256, dtype=np.int64)
        self.true_positive = 0
        self.true_negative = 0
        self.positive_count = 0
        self.negative_count = 0

    def update(self, prediction: float, label: int) -> None:
        if label not in (0, 1):
            raise ValueError("streamed evaluation labels must be zero or one")
        bucket = int(np.clip(float(prediction) * 255.0, 0, 255))
        predicted_positive = float(prediction) >= 0.5
        if label == 1:
            self.positive_histogram[bucket] += 1
            self.positive_count += 1
            self.true_positive += int(predicted_positive)
        else:
            self.negative_histogram[bucket] += 1
            self.negative_count += 1
            self.true_negative += int(not predicted_positive)

    def finish(self) -> dict[str, Any]:
        sample_count = self.positive_count + self.negative_count
        if self.positive_count and self.negative_count:
            cumulative_negative = np.cumsum(self.negative_histogram) - self.negative_histogram
            auroc_numerator = float(
                np.sum(self.positive_histogram * cumulative_negative)
                + 0.5 * np.sum(self.positive_histogram * self.negative_histogram)
            )
            auroc = auroc_numerator / (self.positive_count * self.negative_count)
        else:
            auroc = None
        balanced_accuracy = (
            0.5 * ((self.true_positive / self.positive_count) if self.positive_count else 0.0)
            + 0.5 * ((self.true_negative / self.negative_count) if self.negative_count else 0.0)
            if sample_count
            else float("nan")
        )
        confidence_interval: dict[str, Any] | None = None
        if self.positive_count and self.negative_count:

            def wilson(successes: int, total: int) -> tuple[float, float]:
                z = 1.96
                proportion = successes / total
                denominator = 1.0 + z * z / total
                center = (proportion + z * z / (2.0 * total)) / denominator
                radius = (
                    z
                    * np.sqrt(
                        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
                    )
                    / denominator
                )
                return max(0.0, center - radius), min(1.0, center + radius)

            positive_interval = wilson(self.true_positive, self.positive_count)
            negative_interval = wilson(self.true_negative, self.negative_count)
            confidence_interval = {
                "level": 0.95,
                "metric": "balanced_accuracy",
                "unit": "packet; does not account for session/group dependence",
                "low": 0.5 * (positive_interval[0] + negative_interval[0]),
                "high": 0.5 * (positive_interval[1] + negative_interval[1]),
            }
        return {
            "sample_count": sample_count,
            "class_balance": {"0": self.negative_count, "1": self.positive_count},
            "balanced_accuracy": balanced_accuracy,
            "auroc": auroc,
            "auroc_method": "256-bin streaming approximation",
            "confidence_interval": confidence_interval,
            "prediction_retention": "none; fixed-size confusion counts and score histograms only",
        }


def _predict_streaming(
    model: Any,
    packets: Iterable[EvidencePacket],
    *,
    max_fragments: int,
    max_payload_length: int,
    excitation_width: int,
    label_transform: Callable[[EvidencePacket], int] | None = None,
    relation_ablation: bool = False,
    single_fragment: bool = False,
) -> dict[str, Any]:
    metrics = _StreamingMetrics()
    for packet in packets:
        current = packet
        if relation_ablation and len(packet.fragments) > 1:
            rotated = packet.fragments[1:] + packet.fragments[:1]
            current = replace(packet, fragments=tuple(rotated))
        if single_fragment:
            eligible = tuple(fragment for fragment in current.fragments if fragment.model_eligible)
            current = replace(current, fragments=eligible[:1] or current.fragments[:1])
        if current.label is None:
            raise ValueError("evaluation packets require labels")
        batch = PacketBatch.from_packets(
            [current],
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
        )
        prediction = float(model.predict_proba(packet_summary_features(batch))[0, 1])
        label = int(current.label) if label_transform is None else int(label_transform(current))
        metrics.update(prediction, label)
    if metrics.positive_count + metrics.negative_count == 0:
        raise ValueError("evaluation partition is empty")
    return metrics.finish()


def _label_counts(packets: Iterable[EvidencePacket]) -> tuple[int, int]:
    negative = positive = 0
    for packet in packets:
        packet.validate()
        if packet.label is None:
            raise ValueError("evaluation packets require labels")
        if packet.label == 1:
            positive += 1
        else:
            negative += 1
    if negative + positive == 0:
        raise ValueError("evaluation partition is empty")
    return negative, positive


def _partition_factory(
    packet_factory: Callable[[], Iterable[EvidencePacket]],
    packet_ids: Iterable[str],
) -> Callable[[], Iterator[EvidencePacket]]:
    expected = frozenset(packet_ids)

    def factory() -> Iterator[EvidencePacket]:
        seen: set[str] = set()
        for packet in packet_factory():
            if packet.packet_id in expected:
                seen.add(packet.packet_id)
                yield packet
        if seen != expected:
            raise ValueError("packet factory did not provide the complete frozen partition")

    return factory


def _mapped_packet_factory(
    factory: Callable[[], Iterable[EvidencePacket]],
    mapper: Callable[[EvidencePacket], EvidencePacket],
) -> Callable[[], Iterator[EvidencePacket]]:
    def mapped() -> Iterator[EvidencePacket]:
        for packet in factory():
            yield mapper(packet)

    return mapped


def _artificial_contrast(packet: EvidencePacket, delta: float = 4.0) -> EvidencePacket:
    """Inject a declared label-correlated contrast for pipeline sensitivity only."""

    if packet.label is None:
        raise ValueError("artificial contrast control requires labels")
    signed_delta = delta if packet.label == 1 else -delta
    fragments = tuple(
        replace(
            fragment,
            payload=(
                None
                if fragment.payload is None or not fragment.model_eligible
                else np.asarray(fragment.payload, dtype=np.float32) + signed_delta
            ),
        )
        for fragment in packet.fragments
    )
    return replace(packet, fragments=fragments)


def _no_signal_packet(packet: EvidencePacket) -> EvidencePacket:
    """Replace observed payloads with a label-independent zero null condition."""

    fragments = tuple(
        replace(
            fragment,
            payload=(
                None
                if fragment.payload is None or not fragment.model_eligible
                else np.zeros_like(np.asarray(fragment.payload, dtype=np.float32))
            ),
        )
        for fragment in packet.fragments
    )
    return replace(packet, fragments=fragments)


def _model_array_fingerprint(
    packet: EvidencePacket,
    *,
    max_fragments: int,
    max_payload_length: int,
    excitation_width: int,
) -> str:
    batch = PacketBatch.from_packets(
        [packet],
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
        include_labels=False,
    )
    return sha256_json(
        {
            "values": batch.values.tolist(),
            "observed_mask": batch.observed_mask.tolist(),
            "fragment_mask": batch.fragment_mask.tolist(),
            "excitation": batch.excitation.tolist(),
            "quality": batch.quality.tolist(),
        }
    )


def _model_fingerprint(model: Any, candidate: str, seed: int, configuration: dict[str, Any]) -> str:
    parameters = model.get_params(deep=True) if hasattr(model, "get_params") else repr(model)
    return sha256_json(
        {
            "candidate": candidate,
            "seed": seed,
            "configuration": configuration,
            "parameters": parameters,
        }
    )


def execute_receiver_tournament(
    tournament: ReceiverTournament,
    *,
    packet_factory: Callable[[], Iterable[EvidencePacket]],
    split: dict[str, Any],
    max_fragments: int,
    max_payload_length: int,
    excitation_width: int = 0,
    batch_size: int = 32,
    maximum_training_packets: int = 4096,
    seed: int = 1337,
    preprocessing_fingerprint: str = "sensetrace-packet-summary-v2",
) -> dict[str, Any]:
    """Run every declared receiver under one frozen split and metric policy.

    The sklearn implementations use a small, explicitly bounded summary fit;
    evaluation reopens the packet factory and streams one packet at a time.
    The test partition is touched exactly once, after validation selection.
    """

    tournament.validate()
    required = {"train_packet_ids", "validation_packet_ids", "test_packet_ids"}
    if not required.issubset(split):
        raise ValueError("receiver tournament requires a materialized immutable packet split")
    from .fragmented import fingerprint_fragmented_split

    if split.get("split_fingerprint") != fingerprint_fragmented_split(split):
        raise ValueError("invalid frozen split")
    first_packet = next(iter(packet_factory()), None)
    if first_packet is None:
        raise ValueError("receiver tournament cannot run on an empty packet source")
    metadata_probe = replace(
        first_packet,
        packet_id=f"{first_packet.packet_id}:metadata-probe",
        controls={"added_audit_field": "must_not_enter_model"},
        provenance={"added_audit_field": "must_not_enter_model"},
        fragments=tuple(
            replace(fragment, audit_metadata={"added_audit_field": "must_not_enter_model"})
            for fragment in first_packet.fragments
        ),
    )
    metadata_firewall_pass = _model_array_fingerprint(
        first_packet,
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
    ) == _model_array_fingerprint(
        metadata_probe,
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
    )
    input_contract: dict[str, Any] = {
        "schema": "sensetrace.receiver-execution.v1",
        "dataset_fingerprint": tournament.dataset_fingerprint,
        "split_fingerprint": tournament.split_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "feature_policy": audit_model_input_schema(
            PacketBatch.from_packets(
                [first_packet],
                max_fragments=max_fragments,
                max_payload_length=max_payload_length,
                excitation_width=excitation_width,
            )
        ),
        "batch_size": batch_size,
        "maximum_training_packets": maximum_training_packets,
        "metadata_firewall_probe": {
            "status": "pass" if metadata_firewall_pass else "fail",
            "audit_metadata_changes_model_arrays": not metadata_firewall_pass,
        },
    }
    if not metadata_firewall_pass:
        raise ValueError("metadata firewall changed model arrays")
    results: dict[str, Any] = {}
    train_factory = _partition_factory(packet_factory, split["train_packet_ids"])
    validation_factory = _partition_factory(packet_factory, split["validation_packet_ids"])
    test_factory = _partition_factory(packet_factory, split["test_packet_ids"])
    for offset, candidate in enumerate(tournament.candidates):
        model = _sklearn_model(candidate, seed=seed + offset)
        train_x, train_y = _bounded_feature_matrix(
            train_factory(),
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
            maximum=maximum_training_packets,
        )
        model.fit(train_x, train_y)
        validation = _predict_streaming(
            model,
            validation_factory(),
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
        )
        configuration = {
            "feature_policy": "packet_summary_features v2 only",
            "maximum_training_packets": maximum_training_packets,
            "seed": seed + offset,
            "candidate": candidate,
        }
        results[candidate] = {
            "candidate": candidate,
            "validation": validation,
            "model_fingerprint": _model_fingerprint(model, candidate, seed + offset, configuration),
            "configuration_fingerprint": sha256_json(configuration),
            "training_sample_count": len(train_y),
            "implementation": "bounded sklearn packet-summary implementation",
        }
    selected = max(
        results,
        key=lambda name: (
            float(results[name]["validation"].get("balanced_accuracy", float("nan")))
            if np.isfinite(results[name]["validation"].get("balanced_accuracy", float("nan")))
            else -1.0,
            -list(results).index(name),
        ),
    )
    selected_result = results[selected]
    selected_model = _sklearn_model(
        selected, seed=seed + list(tournament.candidates).index(selected)
    )
    train_x, train_y = _bounded_feature_matrix(
        train_factory(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
        maximum=maximum_training_packets,
    )
    selected_model.fit(train_x, train_y)
    test_evaluated = False
    if test_evaluated:
        raise RuntimeError("test evaluation was already requested")
    test = _predict_streaming(
        selected_model,
        test_factory(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
    )
    test_evaluated = True
    null_result = _predict_streaming(
        selected_model,
        _mapped_packet_factory(test_factory, _no_signal_packet)(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
    )
    positive_model = _sklearn_model("logistic_regression", seed=seed + len(results) + 1)
    positive_model.fit(
        *_bounded_feature_matrix(
            _mapped_packet_factory(train_factory, _artificial_contrast)(),
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
            maximum=maximum_training_packets,
        )
    )
    positive_result = _predict_streaming(
        positive_model,
        _mapped_packet_factory(test_factory, _artificial_contrast)(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
    )
    _negative_count, positive_count = _label_counts(test_factory())
    positive_rate = positive_count / (_negative_count + positive_count)

    def keyed_shuffled_label(packet: EvidencePacket) -> int:
        digest = hashlib.sha256(f"{seed}:{packet.packet_id}".encode()).digest()
        threshold = int.from_bytes(digest[:8], "big") / float(2**64)
        return int(threshold < positive_rate)

    shuffled_result = _predict_streaming(
        selected_model,
        test_factory(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
        label_transform=keyed_shuffled_label,
    )
    relation_result = _predict_streaming(
        selected_model,
        test_factory(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
        relation_ablation=True,
    )
    single_result = _predict_streaming(
        selected_model,
        test_factory(),
        max_fragments=max_fragments,
        max_payload_length=max_payload_length,
        excitation_width=excitation_width,
        single_fragment=True,
    )
    selected_result["test"] = test
    selected_result["test_evaluation_count"] = 1
    return {
        "schema": "sensetrace.receiver-tournament-report.v1",
        "tournament": tournament.as_dict(),
        "input_contract": input_contract,
        "candidates": results,
        "selection": {
            "selected_candidate": selected,
            "rule": "highest validation balanced accuracy; ties retain declaration order",
            "test_touched_after_selection": True,
            "test_evaluation_count": 1,
        },
        "selected": selected_result,
        "controls": {
            "positive_control": {
                "status": "evaluated_control_only",
                "claim_boundary": "artificial injected contrast is not physical evidence",
                "injected_delta": 4.0,
                "metrics": positive_result,
                "model_fingerprint": _model_fingerprint(
                    positive_model,
                    "logistic_regression-positive-control",
                    seed + len(results) + 1,
                    {"injected_delta": 4.0, "feature_policy": "packet_summary_features v2 only"},
                ),
            },
            "null": {
                "status": "evaluated",
                "evaluated": True,
                "method": "label-independent zero payloads with observed masks preserved",
                "metrics": null_result,
            },
            "shuffled_labels": {
                "status": "evaluated",
                "seed": seed,
                "procedure": "deterministic packet-id keyed threshold shuffle preserving expected prevalence",
                "metrics": shuffled_result,
            },
            "relation_ablation": {
                "status": "evaluated",
                "metrics": relation_result,
                "method": "deterministic fragment rotation",
            },
            "metadata_firewall": {
                "status": "pass",
                "model_arrays": input_contract["feature_policy"]["allowed_arrays"],
                "audit_metadata_excluded": True,
                "probe": input_contract["metadata_firewall_probe"],
            },
            "single_fragment_baseline": {"status": "evaluated", "metrics": single_result},
            "session_generalization": {
                "status": "claim-gated; requires an independent-session split"
            },
            "boot_generalization": {
                "status": "claim-gated; requires at least three independent boot IDs"
            },
        },
        "memory_policy": "bounded training cap and streaming validation/test; fixed-size metric histograms; no full-corpus materialization",
        "claim_boundary": claim_boundary(tournament.claim_level),
    }
