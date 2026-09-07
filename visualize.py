import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path
    import treescope

    # Viewing structure needs no GPU. Force CPU so JAX/CUDA init can't
    # segfault the kernel on this WSL box (status 139). Remove to use GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    sys.path.insert(0, str(Path.cwd() / "src"))

    import jax
    import marimo as mo
    from flax import nnx

    return jax, mo, nnx, treescope


@app.cell
def _(mo, nnx):
    from vit_ie.config import Config
    from vit_ie.model import ViT

    try:
        config = Config.from_yaml("config.yaml")
        mc, tc = config.model, config.trainer
        model = ViT(
            img_size=mc.img_size,
            patch_size=mc.patch_size,
            dim=mc.dim,
            depth=mc.depth,
            num_heads=mc.num_heads,
            dropout_rate=mc.dropout_rate,
            rngs=nnx.Rngs(0),
        )
        source = "config.yaml"
    except FileNotFoundError:
        model = ViT(rngs=nnx.Rngs(0))
        source = "ViT defaults"

    mo.vstack(
        [
            mo.md(f"_Sucessfully built model from `{source}`_"),
            mo.accordion(config.to_dict()),
        ]
    )
    return mc, model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ViT for Illuminant Estimation
    """)
    return


@app.cell
def _(mo):
    mo.outline(label="Table of Contents")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Statistics
    """)
    return


@app.cell
def _(jax, mc, mo, model, nnx):
    num_patches = int((mc.img_size // mc.patch_size) ** 2)
    encoder_head_dims = int(mc.dim // mc.num_heads)
    mlp_ratio = 4  # CONSTANT, non-configurable
    mlp_hidden_dims = int(mc.dim * mlp_ratio)

    total_params = sum(
        int(x.size) for x in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    encoder_params = sum(
        int(x.size) for x in jax.tree.leaves(nnx.state(model.blocks, nnx.Param))
    )

    encoder_pct = (encoder_params / total_params) * 100

    mo.hstack(
        [
            mo.stat(
                value=f"{total_params:,}",
                label="Parameters",
                caption=f"total trainable model parameters",
            ),
            mo.stat(
                value=f"{encoder_params:,}",
                label="Encoder Params",
                caption=f"{encoder_pct:.1f}% of params are in the encoder stack",
            ),
            mo.stat(
                value=f"{(total_params * 4) / 1e6:.2f} MB",
                label="Model Size",
                caption="float32 precision",
            ),
        ],
        justify="center",
    )
    return encoder_head_dims, mlp_hidden_dims, mlp_ratio, num_patches


@app.cell(hide_code=True)
def _(encoder_head_dims, mc, mlp_hidden_dims, mlp_ratio, mo, num_patches):
    mo.md(f"""
    ## Architecture

    The Vision Transformer we use contains three key parts:
    1. **Patch embedding**: A ${mc.patch_size}\times{mc.patch_size}$ convolution projecting ${mc.img_size}\times{mc.img_size}$ images into ${mc.dim}$-dim patch embeddings, with a learnable `pos_embed` for ${num_patches}$ patches

    2. **Encoder stack**: {mc.depth} encoder blocks
        - ${mc.num_heads}$-head self-attention (head dims={encoder_head_dims})
        - Feed-forward MLP with ${mlp_hidden_dims}$ hidden units ({mlp_ratio}x expansion)
        - Dropout rate of ${mc.dropout_rate * 100:.0f}\%$

    3. **Post-encoder**: Final `LayerNorm` followed by a linear head projection into 3 $[R, G, B]$ classes
    """)
    return


@app.cell
def _(mo, model, treescope):
    mo.iframe(treescope.render_to_html(model))
    return


if __name__ == "__main__":
    app.run()
