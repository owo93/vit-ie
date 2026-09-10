from collections.abc import Iterator

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset, load_dataset
from datasets import Image as HFImage
from jax import Array, lax, random

# SimpleCube++ PNGs are 16-bit linear raw sensor data,
# not sRGB. Per the Cube++ protocol the black level (~2048) is left unsubtracted
# and the saturation level stays below 16384.
BLACK_LEVEL = 2048.0
WHITE_LEVEL = 16383.0


@jax.jit
def augment(image: Array, key: Array) -> Array:
    """Apply random augmentations to the input image.

    Args:
        image: Input image of shape (H, W, C).
        key: PRNG key

    Returns:
        Augmented image of shape (H, W, C).
    """
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


def decode_image(data: bytes, img_size: int) -> Array:
    """Decode a SimpleCube++ 16-bit PNG into a linear-light RGB image.

    SimpleCube++ PNGs store 16-bit linear raw sensor data with an unsubtracted
    black level of ~2048 and saturation below 16384, so we subtract the black
    level and scale to [0, 1] without applying any sRGB transfer function.

    Args:
        data: Raw PNG bytes from the HuggingFace dataset.
        img_size: Spatial size to resize the image to.

    Returns:
        Linear-light RGB image of shape (img_size, img_size, 3) in [0, 1].
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, "failed to decode PNG"
    assert image.dtype == np.uint16, f"expected 16-bit image, got {image.dtype}"

    image = np.asarray(image, dtype=np.uint16)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image = image[..., ::-1].astype(np.float32)  # BGR -> RGB
    image = (image - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)

    return jnp.asarray(np.clip(image, 0.0, 1.0))


class SimpleCubePPDataset:
    def __init__(self, split: str, seed: int = 42, img_size: int = 224):
        self.split = split
        self.should_augment = split == "train"
        self.rng = random.key(seed)
        self.samples = self._load_split(split)
        self.img_size = img_size

    def _load_split(self, split: str) -> Dataset:
        return load_dataset("owo93/SimpleCubePP", split=split).cast_column(
            "image", HFImage(decode=False)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Array, Array]:
        sample = self.samples[idx]
        image = decode_image(sample["image"]["bytes"], self.img_size)
        illuminant = jnp.array(sample["illuminant"], dtype=jnp.float32)

        return image, illuminant

    def batches(self, batch_size: int, shuffle: bool = True) -> Iterator[tuple[Array, Array]]:
        """Yield batches of (images, illuminants) tuples from dataset.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the dataset before batching.

        Yields:
            Batches of (images, illuminants) tuples.
        """
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
