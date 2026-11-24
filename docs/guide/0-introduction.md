# Introduction

The design principles of the library and a few core concepts are introduced here.  
You can skip this section if you want to start merging right away: [Typical use cases](1-typical-use-cases.md)

## What is sd-mecha

sd-mecha is a PyTorch-based python library for general purpose operations on [state dictionaries](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html).
Some modules of the library can be extended by design. This allows anyone to contribute new models and operations without needing to change the core library.

Historically, model mergers and conversion scripts have been loading all input models in memory before performing any work.
While this approach may seem like a good idea to accelerate merge times, it takes an unreasonable amount of memory for the actual work that needs to be performed.

In contrast to this practice, sd-mecha prioritizes low memory consumption. It can complete complex state dict tasks on low-end systems like:

- converting a large model to a different format using under a gigabyte of memory
- merging 10 large models together into a single one
- allocating more memory for the intermediate results of expensive methods
- sequentially completing multiple state dict tasks

The way sd-mecha does this is by taking advantage of the [safetensors](https://github.com/huggingface/safetensors) format.
This format specifies a light header that lists all the key names, and the location of their corresponding value in the data section and their span.
This information allows to easily turn a safetensors state dict into a stream of tensors.
With this, it becomes possible to handle one key at a time, from loading to saving, without needing to hold all other keys idle in memory at the same time.

While execution speed is not the first priority of sd-mecha, many optimizations are implemented to accelerate merge times.

## Overview of the public API

There are 2 public modules that user code can import and use:

- `sd_mecha`: for recipe operations like composition, merging, serialization, etc.
- `sd_mecha.extensions`: for extending the features of sd-mecha like model configs and merge methods


## Model configs

A model config describes the keys and tensor shapes of a state dict. It can also define aliases and optional keys. A component groups related keys, for example the `vae` in an SDXL checkpoint.

sd-mecha infers the model config from the input keys by default. If several key formats share those keys, pass a config explicitly. This is common with small adapters.

```python
import sd_mecha

base = sd_mecha.model("base.safetensors", config="sdxl-sgm")
```

List the available configs with `sd_mecha.extensions.model_configs.get_all()`. Call `model_configs.resolve(identifier).components()` to inspect the components in one config.

## Built-in checkpoint key formats

| Model | Config IDs |
| --- | --- |
| Stable Diffusion 1.5 | `sd1-ldm`, `sd1-kohya` |
| Stable Diffusion XL | `sdxl-sgm`, `sdxl-kohya`, `sdxl-kohya_but_diffusers`, `sdxl-diffusers_unet_only` |
| Stable Diffusion 3 | `sd3-comfyui` |
| FLUX Schnell and Dev | `flux-flux` |
| Anima | `anima-diffusion_pipe` |
| MOSS-Image | `moss_image-openmoss`, `moss_image-comfyui` |
| Kolors UNet | `kolors-diffusers` |

For Kolors, load the UNet Safetensors file directly. The config covers the standalone UNet and its LyCORIS adapters. See [Blocks Merging (MBW)](1-typical-use-cases.md#blocks-merging-mbw) for the separate block configs.

## Merge spaces

A merge space describes what the values in a recipe represent. The built-in spaces are:

| Space | Values |
| --- | --- |
| `weight` | Model weights or stored adapter factors |
| `delta` | Differences between model weights or between adapter factors, such as `target - base` |
| `param` | Inputs that control an operation, such as block weights or a rank |

`merge()` expects a `weight` result by default. To save a delta, pass `strict_merge_space="delta"`.

## Is sd-mecha for you?

You might find value in using sd-mecha for state dict tasks if you need to:

- operate on very large checkpoints with minimal memory usage
- merge a large number of checkpoints together
- merge multiple variations of a recipe for an ablation study
- experiment with new merge methods
- keep a trace of past attempts and experiments

On the other hand, you may be inclined to look for alternatives if you:

- just want to merge a small number of models using popular merge strategies
- do not know python programming at all
- do not need to experiment extensively

While sd-mecha is a python library without a graphical user interface, there is a sibling [ComfyUI node pack](https://github.com/ljleb/comfy-mecha) that intends to bridge usage for less technical users.

## Usage and design principles

In sd-mecha, every task revolves around *recipe graphs*.
A recipe is a declarative graph that describes how to produce a state dict. `weighted_sum(a, b)` returns a new recipe node that references `a` and `b`.

For example, you can:

- compose recipe graphs together into larger recipe graphs
- materialize a state dict from a recipe graph to disk or memory
- serialize a recipe graph to a human-readable textual format (.mecha) and then deserialize it later

Recipe graphs have a lifecycle that goes something like this:

1. Create or deserialize a recipe graph.
2. Run it with `merge()` or save it with `serialize()`.

The library uses recipe graphs as the planning tool for all state dict operations.
Before loading tensor data, sd-mecha plans which output keys it can compute from the available input keys and each method's key map. This helps it load tensors when needed and release cached intermediate results after their last consumer runs.

Next: [Typical use cases](1-typical-use-cases.md)
