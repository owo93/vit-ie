import jax
from absl import flags

from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.config import Config
from flax_illuminant_estimation.lib.training import run_training

FLAGS = flags.FLAGS


def main() -> None:
    config = Config.from_yaml(FLAGS.config) if FLAGS.config else Config()

    jax.config.update("jax_default_matmul_precision", "high")

    train_ds = SimpleCubePPDataset(
        "train", seed=config.trainer.seed, img_size=config.model.img_size
    )
    test_ds = SimpleCubePPDataset(
        "test", seed=config.trainer.seed + 1, img_size=config.model.img_size
    )

    run_training(config, train_ds, test_ds, sync=FLAGS.sync)


if __name__ == "__main__":
    main()
