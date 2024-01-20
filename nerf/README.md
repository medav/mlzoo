# NeRF: Neural Radiance Fields (2020)

|||
|-|-|
| Original Paper | [arXiv](https://arxiv.org/pdf/2003.08934.pdf) |
| Original Source | [GitHub](https://github.com/bmild/nerf) |
| Reference Datasets | [Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) |
|||

## Getting Started
Download one of the reference scenes from the above Google Drive link.

## Examples
| Example | Description |
|-|-|
| Single Forward | Runs a single forward pass of 65536 rays with 32 sample points. |
|||

How to run:
```bash
# Single Forward Pass
$ python -m nerf.examples.single_fwd
```


## Validation
This code was written based heavily on the original TensorFlow implementation
from the paper's original authors. Numerical validation has not been performed
so this implementation may produce different results.
