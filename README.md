# ML Zoo
Here's a zoo of many machine learning applications!

**Why did I make this?** There's a lot of ML apps written in with different
frameworks and different dependencies. In addition, code quality of many
implementations (even official ones!) is concerning. **My goal with this repo
is to provide concise, well-commented, easy-to-understand implementations of
many importand ML models**

**Guiding Philosophy and Goals for the code in this repo**:
1. **Correctness**: Implement a given application as faithfully as possible to the original source. Note: this does not necessarily mean all the code in this repo is a perfect reproduction / some differences may exist.
2. **Simplicity**: Simplify implementations down to their bare essance -- reducing dependencies as much as possible and rewriting needlessly complex code with an eye toward simplicity.
3. **Educational Value**: Provide concise comments on model architecture and unclear code to make this an educational codebase.
4. **Distillation of Algorithmic and Systems contributions**: Provide (where possible) the ability to generate synthetic data with realistic input shapes for the purpose of enabling systems research without the need for original training data.
5. **Performance Profiling**: Write code in a way that enables easy performance profiling. Specifically, I aim to ensure code works with PyTorch Dynamo where possible.
6. **Modularity**: All apps are written completely independently of eachother to allow any app to be ripped out and used elsewhere.


## Overview of the Zoo

## Levels of Validation
Each of the models in this repository is based on some existing academic
publication presenting that model. The implementations here are intended to
reproduce the original model as faithfully as possible, adopting "default"
hyperparameters if such defaults exist. To provide further information on how
accurate these recreations are, the following table summaraizes what level of
confidence / validation has been achieved.

| Validation Level | Description |
|-|-|
| Level 0 | The model in this repository was written based on only the original academic publication. I.e. No code reference was used (Either was not available or not found). |
| Level 1 | The model in this repository was written based on official code implementation by the original authors. Additional code references may have been used as well. |
| Level 2 | In addition to being based on reference code (Level 1), the model in this repository has been studied to provide similar (eyeball validation) loss values when running on provided reference training data. |
| Level 3 | The model in this repository has been verified to produce numerically exact output given identical input and weight values. |
