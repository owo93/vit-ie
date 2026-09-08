# ViT for Illuminant estimation (vit-ie)

An implementation of Vision Transformer for predicting single illuminant color in images trained on the [SimpleCube++](https://github.com/Visillect/CubePlusPlus) dataset, built with [Flax NNX
](https://flax.readthedocs.io/en/latest/why.html)

## Usage

This [uv](https://docs.astral.sh/uv)-managed Python package exposes an `illum` command for training and inference. There is also a marimo notebook for visualizing the model's architecture at [visualize.py](visualize.py)

Rich outputs are logged to `stdout` and `logging` log messages to `stderr`. Append `2> output.log` to the command to log all messages to a file.

> [!TIP]
> Live tail the logs with `tail -f output.log` in a separate terminal.

## Training

### Flags

- `--config`: path to yaml file with hyperparameters (see [example](config.yaml.example))
- `--sync`: enables live-syncing to W&B

## Inference

To infer illuminant chromaticities on an image:

### Flags

- `--image`: path to input image
- `--checkpoint`: path to model checkpoint to use

## Acknowledgements

- [ ] TODO
