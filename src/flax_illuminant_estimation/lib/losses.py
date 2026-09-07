import optax
from jax import Array

from flax_illuminant_estimation.lib.metrics import reproduction_cosine_similarity


def cosine_distance(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Cosine distance 1 - cos(pred, target), the training objective."""
    return 1.0 - optax.losses.cosine_similarity(pred, target, epsilon=epsilon)


def reproduction_cosine_distance(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Cosine distance 1 - cos(pred, target) in reproduction space, the smooth
    surrogate for reproduction angular error."""
    return 1.0 - reproduction_cosine_similarity(pred, target, epsilon=epsilon)
