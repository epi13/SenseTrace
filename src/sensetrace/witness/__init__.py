"""Experimental kernel/eBPF confounder witness plane."""

from .correlation import correlate_witness
from .models import WitnessEvent, WitnessSession
from .observer import BpftraceWitnessObserver, discover_witness_capabilities

__all__ = [
    "BpftraceWitnessObserver",
    "WitnessEvent",
    "WitnessSession",
    "correlate_witness",
    "discover_witness_capabilities",
]
