from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx
from flax.training import orbax_utils


@dataclass
class CheckpointState:
    graphdef: nnx.GraphDef
    model_state: nnx.State
    epoch: int
    config: dict[str, Any]


_checkpointer = ocp.PyTreeCheckpointer()

CHECKPOINT_INTERVAL = 5


def should_checkpoint(epoch: int, total_epochs: int) -> bool:
    """Save every CHECKPOINT_INTERVAL epochs and always on the final epoch."""
    return epoch % CHECKPOINT_INTERVAL == 0 or epoch == total_epochs


def save(state: CheckpointState, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "graphdef": state.graphdef,
        "model": nnx.to_pure_dict(state.model_state),
        "epoch": state.epoch,
        "config": state.config,
    }
    step_dir = checkpoint_dir.resolve() / f"checkpoint_{int(state.epoch):02}"
    _checkpointer.save(
        step_dir,
        ckpt,
        save_args=orbax_utils.save_args_from_target(ckpt),
        force=True,
        custom_metadata={"epoch": state.epoch, "config": state.config},
    )
    return step_dir


def load(path: Path, target: CheckpointState | None = None) -> CheckpointState:
    path = path.resolve()
    abstract_target: dict[str, Any] | None = None
    if target is not None:
        abstract_target = {
            "graphdef": target.graphdef,
            "model": nnx.to_pure_dict(target.model_state),
            "epoch": target.epoch,
            "config": target.config,
        }
    restored = _checkpointer.restore(path, item=abstract_target)
    model_state = nnx.State(nnx.restore_int_paths(restored["model"]))
    return CheckpointState(
        graphdef=restored["graphdef"],
        model_state=model_state,
        epoch=int(restored["epoch"]),
        config=restored.get("config"),
    )


def list_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []

    return sorted(checkpoint_dir.glob("checkpoint_*"))


def latest(checkpoint_dir: Path) -> Path | None:
    found = list_checkpoints(checkpoint_dir)
    return found[-1] if found else None
