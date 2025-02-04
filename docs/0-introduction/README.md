# Introduction

## What is sd-mecha

sd-mecha is a PyTorch-based python library for general purpose operations on state dictionaries.
Some modules of the library can be extended by design so that arbitrary network architectures can be manipulated in arbitrary ways without needing to change the code of the library.

Typical model mergers and conversion scripts will load all input models in memory before performing any work.
While this may seem like a good idea to accelerate merge times as much as possible, it prevents merging recipes with a very large number of models as input.
Additionally, this approach takes an unreasonable amount of memory for the actual work that needs to be performed.

In contrast to this practice, sd-mecha is designed to prioritize low memory consumption. This enables completing complex state dict tasks on low-end systems,
like merging 10 models together into a single one, or batch converting multiple models to a different format.
While execution speed is not the first priority of sd-mecha, many optimizations are implemented to accelerate merge times -- as long as it does not sacrifice low memory usage.

## Overview of the public API

There are 2 public modules that user code can import and use:
- `sd_mecha`: for recipe operations like composition, merging, serialization, etc.
- `sd_mecha.extensions`: for extending the features of sd-mecha like model architecture, model type, merge methods, etc.


## Is sd-mecha for you?

You might find value in using sd-mecha for merging if you need to:
- operate on a very large model with minimal memory usage
- merge a large number of models together
- convert a great number of models one after the other
- experiment with new merge method ideas
- keep a trace of past attempts and experiments

On the other hand, you may be inclined to look for alternatives if you:
- just want to merge a small number of models using popular merge strategies
- do not know python programming at all
- do not need to experiment extensively

Note that while sd-mecha is a python library without a graphical user interface, there are works in progress that are powered by the library and do not require python knowledge:

- [comfyui nodes](https://github.com/ljleb/comfy-mecha)

## Usage and design principles

In sd-mecha, virtually every feature revolves around "recipes".
A recipe is a list of ordered instructions that explain methodically how to derive a state dict.

For example, you can:
- compose recipes together into larger recipes
- merge a recipe to disk or into an in-memory dictionary
- serialize a recipe to a human-readable format and then deserialize it later

Recipes have a lifecycle. First a recipe is created, and then it can be used in different ways.

The library uses the recipe as a planning tool for intricate merge scenarios, which can then methodically be merged in specific ways later.
Planning state dict operations in advance, when the time comes to materialize a state dict from a recipe, allows the library to pick the best timing to load a tensor from disk or to merge already loaded tensors.

Next: [Recipes](../1-merge-methods)
