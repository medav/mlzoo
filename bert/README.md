# BERT (2018)

|||
|-|-|
| Original Paper | [arXiv](https://arxiv.org/pdf/1810.04805.pdf) |
| Original Source | [GitHub](https://github.com/google-research/bert) |
| Reference Source | [HuggingFace](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) |
|||

## Getting Started
Extracting weights from the reference (requires HF transformers):
```bash
$ python -m bert.ref.extract_weights
```

## Examples
| Example | Description |
|-|-|
| SQuAD Inference | Runs a question-answer inference on a predefined context and question |
| Single Forward | Runs a single forward pass of batch size 1, seqlen 512 through the encoder |
|||

How to run:
```bash
# SQuAD Inference
$ python -m bert.examples.squad_infer
Question: What is the focus of this paper?

My best guess:
isa - aware mapping problem

# Single Forward Pass
$ python -m bert.examples.single_fwd
```


## Validation
This code was validated to be faithful to the reference in `ref/` which is from
the HuggingFace transformers codebase. We are not aware of any differences
between the HF and original Google code but don't discount the possibility there
could be differences.
