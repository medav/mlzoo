# MeshGraphNets (2018)

|||
|-|-|
| Original Paper | [arXiv](https://arxiv.org/abs/2010.03409) |
| Original Source | [GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) |
|||

## Examples

## Validation
This code was written based off the original source in the Deep Mind repository.
Validation includes ensuring numerically correct output is produced for same
weight values.

Notes:
* `InvertableNorm` was created to match the `Normalizer` from the original model
* `unsorted_segsum` implements TensorFlow's UnsortedSegmentSum.
