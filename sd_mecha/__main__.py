import click
import functools
import pathlib
import torch
import traceback
from sd_mecha.recipe_merger import RecipeMerger
from sd_mecha.recipe_serializer import deserialize
from sd_mecha.user_error import UserError
from typing import Optional


DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def except_fallback(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        debug = kwargs['debug']
        try:
            return f(*args, **kwargs)
        except UserError as e:
            print_exception(debug, e)
        except Exception as e:
            print_exception(debug, e)
            if not debug:
                click.echo("An unexpected error occurred. Run with --debug to see more details.", err=True)
    return wrapped


def print_exception(debug: bool, e: Exception):
    if debug:
        click.echo(traceback.format_exc(), err=True)
    else:
        click.echo(str(e), err=True)


@click.group()
def main():
    pass


@click.command(help="Merge models following a recipe")
@click.argument("recipe", type=pathlib.Path)
@click.option("--base-directory", "-b", type=pathlib.Path, default=pathlib.Path(), help="Base directory for .safetensors checkpoints.")
@click.option("--output", "-o", type=Optional[pathlib.Path], default=None, help="Output file for the merge result.")
@click.option("--threads", "-j", default=1, help="Number of keys to merge in parallel.")
@click.option("--device", "-d", default="cpu", help="Device used for merging.")
@click.option("--dtype", "-p", type=click.Choice(list(DTYPE_MAPPING.keys())), default="fp32", help="Work precision.")
@click.option("--save-dtype", "-t", type=click.Choice(list(DTYPE_MAPPING.keys())), default="fp16", help="Save precision.")
@click.option("--debug", is_flag=True, help="Print the stacktrace when an error occurs.")
@except_fallback
def merge(
    recipe: pathlib.Path,
    base_directory: pathlib.Path,
    output: pathlib.Path,
    threads: int,
    device: str,
    dtype: str,
    save_dtype: str,
    debug: bool,
):
    if debug:
        click.echo("Merge in debug mode.", err=True)

    if output is None:
        output = base_directory / "merge.safetensors"

    with open(recipe, "r") as f:
        recipe = deserialize(f.readlines())

    scheduler = RecipeMerger(
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
