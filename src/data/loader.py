from collections.abc import Iterator

import jax
import jax.numpy as jnp
from datasets import Dataset, load_dataset
from jax import Array, lax, random
from PIL import Image


@jax.jit
def augment(image: Array, key: Array) -> Array:
    k1, k2, k3 = random.split(key, 3)
    # flip horizontal
    image = jnp.where(random.bernoulli(k1), jnp.flip(image, axis=1), image)

    # rotate
    k = random.randint(k2, shape=(), minval=0, maxval=4)
    image = lax.switch(
        k,
        [
            lambda x: x,
            lambda x: jnp.rot90(x, k=1, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=2, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=3, axes=(0, 1)),
        ],
        image,
    )

    # flip vertical
    image = jnp.where(random.bernoulli(k3), jnp.flip(image, axis=0), image)

    return image


batched_augment = jax.jit(jax.vmap(augment, in_axes=(0, 0)))


class SimpleCubePPDataset:
    def __init__(self, split: str, seed: int = 42, img_size: int = 224):
        self.split = split
        self.should_augment = split == "train"
        self.rng = random.key(seed)
        self.samples = self._load_split(split)
        self.img_size = img_size

    def _load_split(self, split: str) -> Dataset:
        return load_dataset("owo93/SimpleCubePP", split=split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Array, Array]:
        sample = self.samples[idx]
        image = sample["image"].convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        image = jnp.array(image, dtype=jnp.float32) / 255.0
        illuminant = jnp.array(sample["illuminant"], dtype=jnp.float32)

        return image, illuminant

    def batches(self, batch_size: int, shuffle: bool = True) -> Iterator[tuple[Array, Array]]:
        self.rng, shuffle_key = random.split(self.rng)

        indices = jnp.arange(len(self))
        if shuffle:
            indices = random.permutation(shuffle_key, indices)

        for start_idx in range(0, len(self), batch_size):
            self.rng, augment_key = random.split(self.rng)

            batch_indices = indices[start_idx : start_idx + batch_size]
            if len(batch_indices) < batch_size:
                continue

            images, illuminants = [], []
            for idx in batch_indices:
                image, illuminant = self[int(idx)]
                images.append(image)
                illuminants.append(illuminant)

            images = jnp.stack(images)

            if self.should_augment:
                batch_keys = random.split(augment_key, batch_size)
                images = batched_augment(images, batch_keys)

            yield images, jnp.stack(illuminants)
