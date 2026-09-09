# ruff: noqa

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ViT Illuminant Inference

    Load a trained checkpoint and estimate the scene illuminant chromaticity of an image.
    """)
    return


@app.cell
def _():
    import os
    import io
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    from flax import nnx
    from PIL import Image, ImageDraw

    from vit_ie.checkpoint import latest, list_checkpoints, load
    from vit_ie.config import Config
    from vit_ie.model import ViT

    return (
        Config,
        Image,
        ImageDraw,
        Path,
        ViT,
        io,
        jax,
        jnp,
        latest,
        list_checkpoints,
        load,
        mo,
        nnx,
        np,
    )


@app.cell
def _(jax, mo):
    mo.md(f"""
    Using device: {jax.local_devices()[0].platform}
    """)
    return


@app.cell
def _(Image, ImageDraw, ViT, jnp, np):
    def _preprocess(image: Image.Image, img_size: int) -> jnp.ndarray:
        """Resize and normalize a PIL image into a float32 array."""
        img = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
        return jnp.array(img, dtype=jnp.float32) / 255.0

    def estimate_illuminant(model: ViT, image: Image.Image, img_size: int) -> jnp.ndarray:
        """Predict the illuminant chromaticity vector for a single image.

        Args:
            model: Trained ViT illuminant-estimation model.
            image: RGB image to estimate the illuminant for.
            img_size: Size the model expects its input to be.

        Returns:
            Predicted illuminant chromaticity vector of length 3.
        """
        img_batch = jnp.expand_dims(_preprocess(image, img_size), axis=0)
        pred = model(img_batch, train=False)
        return pred[0]

    def render_result(image: Image.Image, pred: jnp.ndarray) -> Image.Image:
        """Compose input, color-corrected image, swatch, and prediction text.

        Args:
            image: RGB input image.
            pred: Predicted illuminant chromaticity vector of length 3.

        Returns:
            The composited comparison canvas.
        """
        r, g, b = (float(x) for x in pred)

        img_arr = jnp.array(image, dtype=jnp.float32) / 255.0
        illum = jnp.array([r, g, b], dtype=jnp.float32)
        corrected = img_arr / (illum + 1e-8)
        corrected = corrected / (corrected[..., 1:2].max() + 1e-8)
        corrected = jnp.clip(corrected, 0.0, 1.0)

        corrected_img = Image.fromarray((np.array(corrected) * 255.0).astype("uint8"))

        w, h = image.width // 4, image.height // 4
        canvas = Image.new("RGB", (image.width * 2, image.height * 2))
        swatch = Image.new("RGB", (w, h), (int(r * 255), int(g * 255), int(b * 255)))
        canvas.paste(image, (0, 0))
        canvas.paste(swatch, (0, image.height))
        canvas.paste(corrected_img, (image.width, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "Input", fill="white")
        draw.text((image.width + 10, 10), "Corrected (divide by illuminant)", fill="white")
        lines = [
            f"Chromaticity: ({r:.6f}, {g:.6f}, {b:.6f})",
            f"RGB:          ({int(r * 255)}, {int(g * 255)}, {int(b * 255)})",
        ]
        for i, line in enumerate(lines):
            draw.text((image.width + 10, image.height + 10 + i * 20), line, fill="white")

        return canvas

    return estimate_illuminant, render_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load checkpoint
    """)
    return


@app.cell
def _(Config, Path, latest, list_checkpoints, mo):
    ckpt_dir = (
        Config.from_yaml("config.yaml").trainer.checkpoint_dir
        if Path("config.yaml").is_file()
        else Config().trainer.checkpoint_dir
    )
    ckpts = list_checkpoints(ckpt_dir)

    mo.stop(
        not ckpts,
        mo.md(f"**No checkpoints** found in `{ckpt_dir}`. Run `uv run train` first."),
    )

    picker = mo.ui.dropdown(
        options={str(path): path for path in ckpts},
        value=str(latest(ckpt_dir)),
        label="Restore from checkpoint",
        full_width=True,
    )
    picker
    return (picker,)


@app.cell
def _(Path, ViT, load, mo, nnx, picker):
    ckpt_path = Path(picker.value)
    state, meta = load(ckpt_path)
    model_cfg = meta.config["model"]

    model = ViT(
        img_size=int(model_cfg["img_size"]),
        patch_size=int(model_cfg["patch_size"]),
        dim=int(model_cfg["dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
        rngs=nnx.Rngs(0),
    )
    nnx.update(model, state.model_state)

    mo.md(f"_Restored `{ckpt_path}` — epoch {meta.epoch}_")
    return model, model_cfg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimate illuminant
    """)
    return


@app.cell
def _(mo):
    upload = mo.ui.file(
        filetypes=["image/png", "image/jpeg", "image/webp", "image/bmp"],
        label="Upload an image",
        multiple=False,
    )

    mo.vstack([upload])
    return (upload,)


@app.cell
def _(Image, io, mo, upload):
    file = upload.value[0]
    image = Image.open(io.BytesIO(file.contents)).convert("RGB")

    mo.image(image, width="480px")
    return (image,)


@app.cell
def _(estimate_illuminant, image, mo, model, model_cfg, render_result):
    pred = estimate_illuminant(model, image, int(model_cfg["img_size"]))
    canvas = render_result(image, pred)

    r, g, b = (float(x) for x in pred)
    res = {
        "pred": f"{r:.6f}, {g:.6f}, {b:.6f}",
        "rgb": f"{int(r * 255)}, {int(g * 255)}, {int(b * 255)}",
    }

    mo.vstack(
        [
            mo.image(canvas, width="large"),
            mo.md(f"**Chromaticity** {res['pred']}"),
            mo.md(f"**RGB** {res['rgb']}"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
