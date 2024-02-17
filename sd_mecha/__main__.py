import sys
import traceback

import click
import pathlib
import torch
from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha.recipe_serializer import deserialize
from typing import Optional


DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


@click.group()
def main():
    pass


@click.command(
    help="Merge models following a recipe",
)
@click.argument(
    "recipe",
    type=pathlib.Path,
)
@click.option(
    "--base-directory", "-b",
    type=pathlib.Path,
    default=pathlib.Path(),
    help="Base directory containing checkpoints in safetensors format",
)
@click.option(
    "--output", "-o",
    type=Optional[pathlib.Path],
    default=None,
    help="File to stream the merge result into",
)
@click.option(
    "--threads", "-j",
    default=1,
    help="Number of keys to merge in parallel",
)
@click.option(
    "--device", "-d",
    default="cpu",
    help="Default device used to merge",
)
@click.option(
    "--dtype", "-p",
    type=click.Choice(list(DTYPE_MAPPING.keys())),
    default="fp32",
    help="Default work precision",
)
@click.option(
    "--save-dtype", "-t",
    type=click.Choice(list(DTYPE_MAPPING.keys())),
    default="fp16",
    help="Default save precision",
)
def merge(
    recipe: pathlib.Path,
    base_directory: pathlib.Path,
    output: pathlib.Path,
    threads: int,
    device: str,
    dtype: str,
    save_dtype: str,
):
    if output is None:
        output = base_directory / "merge.safetensors"

    with open(recipe, "r") as f:
        try:
            recipe = deserialize(f.readlines())
        except ValueError as e:
            print(e, file=sys.stderr)
            exit(1)

    scheduler = MergeScheduler(
        base_dir=base_directory,
        default_device=device,
        default_dtype=DTYPE_MAPPING[dtype],
    )
    scheduler.merge_and_save(
        recipe,
        output_path=output,
        save_dtype=DTYPE_MAPPING[save_dtype],
        threads=threads
    )


main.add_command(merge)
if __name__ == "__main__":
    main()
