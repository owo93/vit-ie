# ViT for Illuminant estimation (vit-ie)

An implementation of Vision Transformer for predicting single illuminant color in images trained on the [SimpleCube++](https://github.com/Visillect/CubePlusPlus) dataset, built with [Flax NNX](https://flax.readthedocs.io/en/latest/why.html)

## Usage

This [uv](https://docs.astral.sh/uv)-managed Python package exposes a `train` command for training. There are also marimo notebooks: [infer.py](infer.py) for running inference with a trained checkpoint and [visualize.py](visualize.py) for visualizing the model's architecture.

Rich outputs are logged to `stdout` and `logging` log messages to `stderr`. Append `2> output.log` to the command to log all messages to a file.

> [!TIP]
> Live tail the logs with `tail -f output.log` in a separate terminal.

## Training

```bash
uv run train --config config.yaml
```

Append `--sync` to live-sync the run to W&B.

### Flags

- `--config`: path to yaml file with hyperparameters (see [example](config.yaml.example))
- `--sync`: enables live-syncing to W&B

## Inference

Open [infer.py](infer.py) as a marimo notebook:

```bash
uv run marimo edit infer.py
```

Pick a checkpoint from the dropdown (or run training first), then upload an image
or enter a path to it. The notebook rebuilds the model from the checkpoint's
saved configuration and displays the predicted illuminant chromaticity alongside
a color-corrected preview.

## Acknowledgements

- [ ] TODO
