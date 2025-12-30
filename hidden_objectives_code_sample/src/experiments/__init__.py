"""Experiment implementations for hidden objectives research."""

from .experiment_1_scaling import ScalingSurfaceExperiment
from .experiment_2_svd import JointLoRASVDExperiment
from .experiment_3_direction import ConcealmentDirectionExperiment
from .experiment_4_layerwise import LayerwiseLocalizationExperiment
from .experiment_5_causal_tracing import ActivationPatchingExperiment
from .experiment_6_causal_transfer import CausalInterventionTransferExperiment

__all__ = [
    "ScalingSurfaceExperiment",
    "JointLoRASVDExperiment",
    "ConcealmentDirectionExperiment",
    "LayerwiseLocalizationExperiment",
    "ActivationPatchingExperiment",
    "CausalInterventionTransferExperiment",
]

