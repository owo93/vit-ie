import optax
from jax import Array

from vit_ie.lib.metrics import reproduction_cosine_similarity


def cosine_distance(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Cosine distance 1 - cos(pred, target), the training objective.

    Args:
        pred: Predicted illuminant chromaticities of shape (..., 3).
        target: Ground-truth illuminant chromaticities of shape (..., 3).
        epsilon: Small constant for numerical stability.

    Returns:
        Per-image cosine distance of shape (...).
    """
    return 1.0 - optax.losses.cosine_similarity(pred, target, epsilon=epsilon)


def reproduction_cosine_distance(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Cosine distance in reproduction space, a smooth surrogate for reproduction angular error.

    Args:
        pred: Predicted illuminant chromaticities of shape (..., 3).
        target: Ground-truth illuminant chromaticities of shape (..., 3).
        epsilon: Small constant for numerical stability.

    Returns:
        Per-image reproduction cosine distance of shape (...).
    """
    return 1.0 - reproduction_cosine_similarity(pred, target, epsilon=epsilon)
