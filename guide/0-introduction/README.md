# Introduction

## What is sd-mecha

sd-mecha is a PyTorch-based python library for general purpose operations on [state dictionaries](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html).
Some modules of the library can be extended by design. This allows anyone to contribute new models and operations without needing to change the core library.

Historically, model mergers and conversion scripts have been loading all input models in memory before performing any work.
While this approach may seem like a good idea to accelerate merge times, it takes an unreasonable amount of memory for the actual work that needs to be performed.

In contrast to this practice, sd-mecha prioritizes low memory consumption. It can complete complex state dict tasks on low-end systems like:
- converting the keys of a model to a different format only using a few megabytes of memory
- merging 10 models together into a single one
- allocating more memory to the intermediate results of expensive methods
- sequentially completing multiple state dict tasks

The way sd-mecha does this is by taking advantage of the [safetensors](https://github.com/huggingface/safetensors) format.
This format specifies a light header that lists all the key names, and the location of their corresponding value in the data section and their span.
This information allows to easily turn a safetensors state dict into a stream of keys.
With this, it becomes possible to handle one key at a time, from loading to saving, without needing to hold all other keys idle in memory at the same time.

While execution speed is not the first priority of sd-mecha, many optimizations are implemented to accelerate merge times.

## Overview of the public API

There are 2 public modules that user code can import and use:
- `sd_mecha`: for recipe operations like composition, merging, serialization, etc.
- `sd_mecha.extensions`: for extending the features of sd-mecha like model configs and merge methods


## Is sd-mecha for you?

You might find value in using sd-mecha for state dict tasks if you need to:
- operate on a very large state dict with minimal memory usage
- merge a large number of models together
- merge multiple variants of a state dict recipe for an ablation study
- experiment with new merge methods
- keep a trace of past attempts and experiments

On the other hand, you may be inclined to look for alternatives if you:
- just want to merge a small number of models using popular merge strategies
- do not know python programming at all
- do not need to experiment extensively

While sd-mecha is a python library without a graphical user interface, there is a work in progress [ComfyUI node pack](https://github.com/ljleb/comfy-mecha) that intends to bridge usage for less technical users.

## Usage and design principles

In sd-mecha, every task revolves around *recipe graphs*.
A recipe graph is a graph of instructions that explain methodically how to derive a state dict.

For example, you can:
- compose recipe graphs together into larger recipe graphs
- materialize a state dict from a recipe graph to disk or memory
- serialize a recipe graph to a human-readable textual format (.mecha) and then deserialize it later

Recipe graphs have a lifecycle that goes something like this:
1. create or deserialize a recipe graph
2. merge and/or serialize the recipe graph

The library uses recipe graphs as the planning tool for all state dict operations.
Planning state dict operations in advance allows to pick the best timing to load tensors from disk and to reuse already loaded tensors.

Next: [Typical use cases](../1-typical-use-cases)
