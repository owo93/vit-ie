from dataclasses import asdict, dataclass
from typing import Any, cast

from etils.epath import Path
from flax import nnx
from orbax.checkpoint import v1 as ocp


@dataclass
class CheckpointState:
    model_state: nnx.State


@dataclass
class CheckpointMetadata:
    epoch: int
    config: dict[str, Any]


CHECKPOINT_INTERVAL = 5


def should_checkpoint(epoch: int, total_epochs: int) -> bool:
    """Save every CHECKPOINT_INTERVAL epochs and always on the final epoch.

    Args:
        epoch: Completed epoch index.
        total_epochs: Total number of training epochs.

    Returns:
        Whether a checkpoint should be written for this epoch.
    """
    return epoch % CHECKPOINT_INTERVAL == 0 or epoch == total_epochs


def save(ckpt: CheckpointState, metadata: CheckpointMetadata, checkpoint_dir: Path) -> Path:
    """Persist a checkpoint state to disk.

    Args:
        ckpt: Checkpoint state to save.
        metadata: Metadata to save alongside the checkpoint.
        checkpoint_dir: Directory in which to write the checkpoint.

    Returns:
        Path to the written checkpoint directory.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step_dir = checkpoint_dir.resolve() / f"checkpoint_{int(metadata.epoch):02}"

    checkpointables = {
        "model": nnx.to_pure_dict(ckpt.model_state),
    }

    ocp.save_checkpointables(
        step_dir, checkpointables, overwrite=True, custom_metadata=asdict(metadata)
    )

    return step_dir


def load(path: Path) -> tuple[CheckpointState, CheckpointMetadata]:
    """Restore a checkpoint state from disk.

    Args:
        path: Path to the checkpoint directory to restore.

    Returns:
        Tuple of the restored model state and the metadata saved with it.

    Raises:
        ValueError: If the checkpoint metadata is not a dictionary.
    """
    path = path.resolve()
    restored_state = ocp.load_checkpointables(
        path,
        {
            "model": None,
        },
    )

    restored_meta = ocp.checkpointables_metadata(path).custom_metadata
    if not isinstance(restored_meta, dict):
        raise ValueError(f"Checkpoint at {path} has unexpected metadata: {restored_meta!r}")
    metadata = cast(dict[str, Any], restored_meta)

    return (
        CheckpointState(
            model_state=nnx.State(nnx.restore_int_paths(restored_state["model"])),
        ),
        CheckpointMetadata(
            epoch=metadata["epoch"],
            config=cast(dict[str, Any], metadata["config"]),
        ),
    )


def list_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """List checkpoint directories, sorted by name.

    Args:
        checkpoint_dir: Directory to scan for checkpoints.

    Returns:
        List of checkpoint paths, sorted by name.
    """
    if not checkpoint_dir.exists():
        return []

    return sorted(checkpoint_dir.glob("checkpoint_*"))


def latest(checkpoint_dir: Path) -> Path | None:
    """Return the most recent checkpoint, or None if no checkpoints exist.

    Args:
        checkpoint_dir: Directory to scan for checkpoints.

    Returns:
        Path of the newest checkpoint, or None if there are none.
    """
    found = list_checkpoints(checkpoint_dir)
    return found[-1] if found else None
