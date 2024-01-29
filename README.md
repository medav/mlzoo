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
