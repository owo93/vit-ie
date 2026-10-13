import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jax import Array
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.pretty import pprint
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

import wandb
from vit_ie.checkpoint import CheckpointMetadata, CheckpointState, save, should_checkpoint
from vit_ie.config import Config
from vit_ie.lib.metrics import (
    create_eval_metrics,
    create_train_metrics,
    error_statistics,
)
from vit_ie.lib.trainer import Trainer, TrainState, eval_step, train_step
from vit_ie.model import ViT

if TYPE_CHECKING:
    from data.loader import SimpleCubePPDataset

ERROR_STAT_FIELDS: tuple[str, ...] = (
    "mean",
    "median",
    "trimean",
    "best_25",
    "worst_25",
    "worst",
)


def _wandb_error_stats(stats: dict[str, float], namespace: str) -> dict[str, float]:
    return {f"{namespace}/{field}": stats[field] for field in ERROR_STAT_FIELDS}


def run_training(
    config: Config,
    train_ds: SimpleCubePPDataset,
    test_ds: SimpleCubePPDataset,
    sync: bool,
) -> None:
    """Train and periodically evaluate a ViT illuminant-estimation model.

    Args:
        config: Full training configuration.
        train_ds: Dataset yielding (images, illuminants) training batches.
        test_ds: Dataset yielding (images, illuminants) test batches.
        sync: Whether to run wandb in online mode.
    """
    train_steps = math.ceil(len(train_ds) / config.trainer.batch_size)
    eval_steps = math.ceil(len(test_ds) / config.trainer.batch_size)
    total_steps = config.trainer.epochs * train_steps

    rngs = nnx.Rngs(config.trainer.seed)
    model = ViT(
        img_size=config.model.img_size,
        patch_size=config.model.patch_size,
        dim=config.model.dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        rngs=rngs,
    )

    trainer = Trainer(config.trainer)
    state: TrainState = trainer.create_train_state(model, train_steps)

    wandb.init(
        project="flax-illuminant-estimation",
        group=config.run.wandb_group,
        tags=config.run.wandb_tags,
        config=config.to_dict() | {"total_steps": total_steps},
        mode="online" if sync else "offline",
        settings=wandb.Settings(console="off"),
    )
    wandb.define_metric("step/*", step_metric="step/global")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")
    wandb.define_metric("iec/*", step_metric="epoch")
    wandb.define_metric("repro/*", step_metric="epoch")
    wandb.define_metric("csce/*", step_metric="epoch")
    pprint(config.to_dict(), expand_all=True, indent_guides=False)

    console = Console()
    table = Table(
        title="Training Progress",
        expand=True,
        row_styles=["", "dim"],
        box=box.SIMPLE,
        min_width=120,
    )
    table.add_column("epoch", justify="right", style="on black")
    table.add_column("train_loss")
    table.add_column("train_ae", style="cyan")
    table.add_column("eval_loss")
    table.add_column("eval_ae", style="cyan")
    table.add_column("eval_rae")
    table.add_column("eval_csce")

    progress = Progress(
        MofNCompleteColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    train_task = progress.add_task("Train", total=train_steps)
    eval_task = progress.add_task("Evaluation", total=eval_steps)
    train_metrics = create_train_metrics()
    eval_metrics = create_eval_metrics()

    with Live(Group(table, progress), console=console, refresh_per_second=4):
        for epoch in range(0, config.trainer.epochs):
            train_metrics.reset()
            eval_metrics.reset()
            progress.reset(train_task)
            progress.reset(eval_task)
            progress.update(
                train_task,
                description=f"epoch [u bold green]{epoch + 1}/{config.trainer.epochs}: training...",
            )
            progress.update(
                eval_task,
                description="[red]evaluating...",
            )

            # Training
            for i, (images, illuminants) in enumerate(
                train_ds.batches(config.trainer.batch_size, drop_last=True)
            ):
                step: dict[str, Array] = train_step(
                    state, model, images, illuminants, config.trainer.dtype
                )
                train_metrics.update(loss=step["train/loss"], angular_error=step["train/ae"])

                current_metrics = train_metrics.compute()

                if i % 5 == 0:
                    wandb.log(
                        {
                            "step/global": state.step.value,
                            "step/loss": float(jnp.mean(step["train/loss"])),
                            "step/lr": float(step["train/lr"]),
                        },
                    )

                progress.update(
                    train_task,
                    advance=1,
                    # ruff: noqa: E501
                    description=f"epoch {epoch + 1}/{config.trainer.epochs}: train loss \u2192 [i bold cyan]{float(current_metrics['loss']):.7f}",
                )

            # Evaluation
            angular_error_batches, reproduction_error_batches = [], []
            corrected_scene_error_batches = []

            for images, illuminants in test_ds.batches(
                config.trainer.batch_size, shuffle=False, drop_last=False
            ):
                step: dict[str, Array] = eval_step(model, images, illuminants, config.trainer.dtype)
                angular_error_batches.append(step["eval/ae"])
                reproduction_error_batches.append(step["eval/rae"])
                corrected_scene_error_batches.append(step["eval/csce"])
                eval_metrics.update(
                    loss=step["eval/loss"],
                    angular_error=step["eval/ae"],
                    reproduction_angular_error=step["eval/rae"],
                    corrected_scene_chromaticity_error=step["eval/csce"],
                )

                current_metrics = eval_metrics.compute()

                progress.update(
                    eval_task,
                    advance=1,
                    # ruff: noqa: E501
                    description=f"epoch {epoch + 1}/{config.trainer.epochs}: eval loss \u2192 [i bold magenta]{float(current_metrics['loss']):.7f}",
                )

            # Full distribution stats
            all_angular_errors = jnp.concatenate(angular_error_batches, axis=0)
            all_reproduction_errors = jnp.concatenate(reproduction_error_batches, axis=0)
            all_corrected_scene_errors = jnp.concatenate(corrected_scene_error_batches, axis=0)
            angular_stats = error_statistics(all_angular_errors)
            reproduction_stats = error_statistics(all_reproduction_errors)
            corrected_scene_stats = error_statistics(all_corrected_scene_errors)

            train_m = train_metrics.compute()
            eval_m = eval_metrics.compute()

            if should_checkpoint(epoch + 1, config.trainer.epochs):
                _, model_state = nnx.split(model)
                metadata = CheckpointMetadata(
                    epoch=epoch + 1,
                    config=config.to_dict(),
                )
                ckpt = CheckpointState(
                    model_state=model_state,
                )
                save(ckpt, metadata, config.trainer.checkpoint_dir)

            wandb.log(
                {
                    "train/loss": float(train_m["loss"]),
                    "train/ae": float(train_m["angular_error"]),
                    "eval/loss": float(eval_m["loss"]),
                    "eval/ae": float(eval_m["angular_error"]),
                    "eval/rae": float(eval_m["reproduction_angular_error"]),
                    "eval/csce": float(eval_m["corrected_scene_chromaticity_error"]),
                    **_wandb_error_stats(angular_stats, "iec"),
                    **_wandb_error_stats(reproduction_stats, "repro"),
                    **_wandb_error_stats(corrected_scene_stats, "csce"),
                    "epoch": epoch + 1,
                }
            )

            table.add_row(
                f"{str(epoch + 1).zfill(2)}",
                f"{float(train_m['loss']):.6f}",
                f"{float(train_m['angular_error']):.3f}\xb0",
                f"{float(eval_m['loss']):.6f}",
                f"{float(eval_m['angular_error']):.3f}\xb0",
                f"{float(eval_m['reproduction_angular_error']):.3f}\xb0",
                f"{float(eval_m['corrected_scene_chromaticity_error']):.3f}\xb0",
            )

        wandb.finish()
