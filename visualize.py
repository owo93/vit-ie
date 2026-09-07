import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    # Viewing structure needs no GPU. Force CPU so JAX/CUDA init can't
    # segfault the kernel on this WSL box (status 139). Remove to use GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    sys.path.insert(0, str(Path.cwd() / "src"))

    import jax
    import marimo as mo
    from flax import nnx

    return jax, mo, nnx


@app.cell
def _(mo, nnx):
    from vit_ie.config import Config
    from vit_ie.model import ViT

    try:
        mc = Config.from_yaml("config.yaml").model
        model = ViT(
            img_size=mc.img_size,
            patch_size=mc.patch_size,
            dim=mc.dim,
            depth=mc.depth,
            num_heads=mc.num_heads,
            dropout_rate=mc.dropout_rate,
            rngs=nnx.Rngs(0),
        )
        source = f"config.yaml (patch {mc.patch_size}, dim {mc.dim}, depth {mc.depth}, heads {mc.num_heads})"
    except FileNotFoundError:
        model = ViT(rngs=nnx.Rngs(0))
        source = "ViT defaults"

    mo.md(f"Built from **{source}**.")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ViT Architecture
    """)
    return


@app.cell
def _(mo, model):
    mo.vstack(
        [
            mo.inspect(model.patch_embed),
            mo.inspect(model.blocks),
            mo.inspect(model.norm),
            mo.inspect(model.head),
        ]
    )
    return


@app.cell
def _(jax, mo, model, nnx):
    total = sum(int(x.size) for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    size_bytes = total * 4

    mo.md(f"**Trainable params:** {total:,}\n**Size (float32):** {size_bytes / 1e6:.2f} MB")
    return


if __name__ == "__main__":
    app.run()
