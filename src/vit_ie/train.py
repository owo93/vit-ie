import jax
from absl import flags

from data.loader import SimpleCubePPDataset
from vit_ie.config import Config
from vit_ie.lib.training import run_training

FLAGS = flags.FLAGS


def main() -> None:
    """Load the config and datasets, then run the training loop."""
    config = Config.from_yaml(FLAGS.config) if FLAGS.config else Config()

    jax.config.update("jax_optimization_level", "O1")
    jax.config.update("jax_default_matmul_precision", "highest")

    train_ds = SimpleCubePPDataset(
        "train", seed=config.trainer.seed, img_size=config.model.img_size
    )
    test_ds = SimpleCubePPDataset(
        "test", seed=config.trainer.seed + 1, img_size=config.model.img_size
    )

    run_training(config, train_ds, test_ds, sync=FLAGS.sync)


if __name__ == "__main__":
    main()
