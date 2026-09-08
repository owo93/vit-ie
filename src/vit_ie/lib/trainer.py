import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from jax.typing import DTypeLike

from vit_ie.config import TrainerConfig
from vit_ie.lib.losses import cosine_distance, reproduction_cosine_distance
from vit_ie.lib.metrics import (
    angular_error,
    corrected_scene_chromaticity_error,
    reproduction_angular_error,
)
from vit_ie.model import ViT

WARMUP_EPOCHS = 3
AUX_WARMUP_EPOCHS = 12


# subclass nnx.Optimizer to directly call .update()
class TrainState(nnx.Optimizer):
    def __init__(
        self,
        model: ViT,
        tx: optax.GradientTransformation,
        schedule: optax.Schedule,
        warmup_steps: int,
        aux_warmup_steps: int,
    ):
        super().__init__(model, tx, wrt=nnx.Param)
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.aux_warmup_steps = aux_warmup_steps
        self.model = model

    @property
    def lr(self) -> Array:
        """Return the current learning rate at the present optimizer step."""
        return jnp.asarray(self.schedule(self.step.value))


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

    def create_schedule(self, epochs: int, peak_lr: float, steps_per_epoch: int) -> optax.Schedule:
        """Create a warmup-then-cosine-decay learning rate schedule.

        Args:
            epochs: Total number of training epochs.
            peak_lr: Peak learning rate reached after warmup.
            steps_per_epoch: Number of optimizer steps per epoch.

        Returns:
            A schedule mapping optimizer step index to a learning rate.
        """
        warmup_steps = WARMUP_EPOCHS * steps_per_epoch

        total_steps = epochs * steps_per_epoch

        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=peak_lr * 0.01,
        )

    def create_train_state(self, model: ViT, steps_per_epoch: int) -> TrainState:
        """Build the optimizer and train state for the given model.

        Args:
            model: The vision transformer to train.
            steps_per_epoch: Number of optimizer steps per epoch.

        Returns:
            A TrainState wrapping the model and its optimizer.
        """
        config: TrainerConfig = self.config
        schedule = self.create_schedule(config.epochs, config.learning_rate, steps_per_epoch)

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=config.weight_decay),
        )

        warmup_steps = WARMUP_EPOCHS * steps_per_epoch
        aux_warmup_steps = AUX_WARMUP_EPOCHS * steps_per_epoch
        return TrainState(model, tx, schedule, warmup_steps, aux_warmup_steps)


@nnx.jit(static_argnames=("dtype",))
def train_step(
    state: TrainState,
    model: ViT,
    images: Array,
    illuminants: Array,
    dtype: DTypeLike,
) -> dict[str, Array]:
    """Run a single training step and return per-batch metrics.

    Args:
        state: Train state providing optimizer, schedule and step counter.
        model: The model being trained.
        images: Image batch of shape (B, H, W, 3).
        illuminants: Ground-truth illuminant chromaticities of shape (B, 3).
        dtype: Compute dtype used for the forward pass.

    Returns:
        Dict mapping metric names to per-batch values.
    """

    def loss_fn(model: ViT) -> tuple[Array, Array]:
        pred = model(images.astype(dtype), train=True).astype(jnp.float32)
        target = illuminants.astype(jnp.float32)

        lam = jnp.clip(state.step.value / state.aux_warmup_steps, 0.0, 1.0)
        loss = jnp.mean(cosine_distance(pred, target)) + lam * jnp.mean(
            reproduction_cosine_distance(pred, target)
        )
        angular_errors = angular_error(pred, target)

        return loss, angular_errors

    (loss, angular_errors), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    state.update(model, grads)
    return {
        "train/loss": loss,
        "train/ae": jnp.degrees(angular_errors),  # (B, )
        "train/lr": state.lr,
    }


@nnx.jit(static_argnames=("dtype",))
def eval_step(
    model: ViT,
    images: Array,
    illuminants: Array,
    dtype: DTypeLike,
) -> dict[str, Array]:
    """Run a single evaluation step and return per-batch metrics.

    Args:
        model: The model being evaluated.
        images: Image batch of shape (B, H, W, 3).
        illuminants: Ground-truth illuminant chromaticities of shape (B, 3).
        dtype: Compute dtype used for the forward pass.

    Returns:
        Dict mapping metric names to per-batch values.
    """
    images = images.astype(jnp.float32)
    pred = model(images.astype(dtype), train=False).astype(jnp.float32)
    target = illuminants.astype(jnp.float32)

    loss = cosine_distance(pred, target)  # (B, )
    angular_errors = angular_error(pred, target)
    reproduction_errors = reproduction_angular_error(pred, target)
    corrected_scene_errors = corrected_scene_chromaticity_error(images, pred, target)

    return {
        "eval/loss": loss,
        "eval/ae": jnp.degrees(angular_errors),
        "eval/rae": jnp.degrees(reproduction_errors),
        "eval/csce": jnp.degrees(corrected_scene_errors),
    }
