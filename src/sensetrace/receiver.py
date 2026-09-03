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
    ]
    return np.column_stack(summaries).astype(np.float32, copy=False)


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
        self, packets: Iterable[EvidencePacket], *, dataset_fingerprint: str
    ) -> NoiseResidualizer:
        if not dataset_fingerprint:
            raise ValueError("noise model requires the source dataset fingerprint")
        self.source_dataset_fingerprint = dataset_fingerprint
        for packet in packets:
            packet.validate()
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
