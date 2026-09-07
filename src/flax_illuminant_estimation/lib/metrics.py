import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array


def angular_error(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Per-image angle (radians) between predicted and true illuminants."""
    cos_sim = optax.losses.cosine_similarity(pred, target, epsilon=epsilon)
    return jnp.arccos(jnp.clip(cos_sim, -1.0 + epsilon, 1.0 - epsilon))


# https://doi.org/10.1109/TPAMI.2016.2582171
def reproduction_cosine_similarity(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Per-image cosine similarity between the residual color cast target / pred
    and neutral gray."""
    ratio = target / jnp.maximum(pred, epsilon)
    neutral = jnp.ones_like(ratio)

    return optax.losses.cosine_similarity(ratio, neutral, epsilon=epsilon)


def reproduction_angular_error(pred: Array, target: Array, epsilon: float = 1e-8) -> Array:
    """Per-image angle (radians) from the reproduction cosine similarity."""
    cos_sim = reproduction_cosine_similarity(pred, target, epsilon)
    return jnp.arccos(jnp.clip(cos_sim, -1.0 + epsilon, 1.0 - epsilon))


def corrected_scene_chromaticity_error(
    image: Array, pred: Array, target: Array, epsilon: float = 1e-8
) -> Array:
    """Per-image angle (radians) between the scene chromaticities of an image
    corrected by the predicted vs. the ground-truth illuminant."""
    corrected_pred = jnp.sum(image / (pred[:, None, None, :] + epsilon), axis=(1, 2))
    corrected_target = jnp.sum(image / (target[:, None, None, :] + epsilon), axis=(1, 2))

    cos_sim = optax.losses.cosine_similarity(corrected_pred, corrected_target, epsilon=epsilon)
    return jnp.arccos(jnp.clip(cos_sim, -1.0 + epsilon, 1.0 - epsilon))


def error_statistics(errors: Array) -> dict[str, float]:
    """Summarize a per-image error distribution (mean/median/quartile/...)."""
    n = len(errors)
    sorted_errors = jnp.sort(errors)
    lower_quartile = float(jnp.percentile(errors, 25).squeeze())
    median = float(jnp.percentile(errors, 50).squeeze())
    upper_quartile = float(jnp.percentile(errors, 75).squeeze())
    return {
        "mean": float(jnp.mean(errors)),
        "median": median,
        "trimean": 0.25 * lower_quartile + 0.5 * median + 0.25 * upper_quartile,
        "best_25": float(jnp.mean(sorted_errors[: n // 4])),
        "worst_25": float(jnp.mean(sorted_errors[n - n // 4 :])),
        "worst": float(sorted_errors[-1].squeeze()),
    }


def create_train_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        angular_error=nnx.metrics.Average("angular_error"),
    )


def create_eval_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        angular_error=nnx.metrics.Average("angular_error"),
        reproduction_angular_error=nnx.metrics.Average("reproduction_angular_error"),
        corrected_scene_chromaticity_error=nnx.metrics.Average(
            "corrected_scene_chromaticity_error"
        ),
    )
