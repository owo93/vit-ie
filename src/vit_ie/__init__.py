import os

import jax
from absl import app, flags, logging

from . import infer, train

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=2 "
    "--xla_gpu_enable_async_all_reduce=true "
    "--xla_gpu_deterministic_reductions=false "
    "--xla_gpu_enable_async_all_gather=true "
    "--xla_gpu_max_kernel_unroll=32"
)

FLAGS = flags.FLAGS
logging.set_verbosity(logging.INFO)

flags.DEFINE_enum("command", None, ["train", "infer"], "command to run")

# Train
flags.DEFINE_string("config", None, "path to config.yaml")
flags.DEFINE_bool("sync", False, "wandb run mode, defaults to offline")

# Infer
flags.DEFINE_string("image", None, "path to input image")
flags.DEFINE_string("checkpoint", None, "path to save checkpoint")


def main(argv: list[str]) -> None:
    """Dispatch to the train or infer command based on argv.

    Args:
        argv: Command-line arguments after the program name.

    Raises:
        ValueError: If no command is given or --image is missing for infer.
    """
    if len(argv) < 1:
        raise ValueError("No command specified. Use --command to specify train or infer.")

    logging.info(f"found device: {jax.local_devices()[0].platform}")

    if argv[1] == "train":
        train.main()
    elif argv[1] == "infer":
        if FLAGS.image is None:
            raise ValueError("--image must be specified for infer command")
        infer.main()


# ruff: noqa: ANN201
def run():
    """Console-script entry point that delegates to absl.app.run."""
    app.run(main)
