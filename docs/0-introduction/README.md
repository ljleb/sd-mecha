# Introduction

## What is sd-mecha

sd-mecha is a PyTorch-based python library for general purpose model merging.
The extension API is configurable by design so that any PyTorch neural network model can be used.

Typical model mergers will load all input models in memory before performing a merge.
While this may seem a good idea to accelerate merge times as much as possible, it prevents merging recipes with a very large number of models as input.

In contrast to this practice, sd-mecha is designed to prioritize low memory consumption to enable completing very complex recipes on user-grade hardware in one go.
While execution speed is not a priority, many optimizations are implemented to accelerate merge times, as long as it does not sacrifice low memory usage.

## Overview of the public API

There are 2 public modules that user code can import and use:
- `sd_mecha`: for recipe operations like composition, merging, serialization, etc.
- `sd_mecha.extensions`: for extending the features of sd-mecha like model architecture, model type, hypers, etc.


## Is sd-mecha for you?

You might find value in using sd-mecha for merging if you need to:
- merge a large number of models together
- merge a series of recipes in one click
- experiment with new merge method ideas
- keep a trace of past attempts and experiments

On the other hand, you may be inclined to look for alternatives if you:
- just want to merge a small number of models using popular merge strategies
- do not know python programming at all
- do not need to experiment extensively

Note that while sd-mecha is a python library without a graphical user interface, there are works in progress that are powered by the library and do not require python knowledge:

- [comfyui nodes](https://github.com/ljleb/comfy-mecha)
- [invokeai nodes](https://github.com/ljleb/invokeai-mecha)

## Usage and design principles

In sd-mecha, every operation revolves around recipes.
A recipe is a list of ordered instructions that explain methodically how to complete a merge.

For example, you can:
- compose recipes together
- merge a recipe to disk or into an in-memory state dict
- serialize a recipe to a human-readable format and then deserialize it later

A recipe has a lifecycle. First a recipe is created or deserialized, and then it can be merged or serialized.
The point of using recipes for everything is that they serve as a planning tool for intricate merge scenarios, which we can then methodically merge in the specific way we want later.
Planning recipes in advance also allows effective merge code to pick the best timing to load a tensor from disk or merge already loaded tensors.

Next: [Recipes](../1-recipes)
