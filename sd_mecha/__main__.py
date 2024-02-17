import click
import functools
import pathlib
import torch
import traceback
from sd_mecha.merge_scheduler import MergeScheduler
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
        verbose = kwargs['verbose']
        try:
            return f(*args, **kwargs)
        except UserError as e:
            print_exception(verbose, e)
        except Exception as e:
            print_exception(verbose, e)
            if not verbose:
                click.echo("An unexpected error occurred. Run with --verbose to see more details.", err=True)
    return wrapped


def print_exception(verbose: bool, e: Exception):
    """Prints exception information based on verbosity."""
    if verbose:
        click.echo(traceback.format_exc(), err=True)
    else:
        click.echo(str(e), err=True)


@click.group()
def main():
    """Main entry point for CLI."""
    pass


@click.command(help="Merge models following a recipe")
@click.argument("recipe", type=pathlib.Path)
@click.option("--base-directory", "-b", type=pathlib.Path, default=pathlib.Path(), help="Base directory for checkpoints.")
@click.option("--output", "-o", type=Optional[pathlib.Path], default=None, help="Output file for the merge result.")
@click.option("--threads", "-j", default=1, help="Number of keys to merge in parallel.")
@click.option("--device", "-d", default="cpu", help="Device used for merging.")
@click.option("--dtype", "-p", type=click.Choice(list(DTYPE_MAPPING.keys())), default="fp32", help="Work precision.")
@click.option("--save-dtype", "-t", type=click.Choice(list(DTYPE_MAPPING.keys())), default="fp16", help="Save precision.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output to show stack trace on error.")
@except_fallback
def merge(
    recipe: pathlib.Path,
    base_directory: pathlib.Path,
    output: pathlib.Path,
    threads: int,
    device: str,
    dtype: str,
    save_dtype: str,
    _verbose: bool,
):
    if output is None:
        output = base_directory / "merge.safetensors"

    with open(recipe, "r") as f:
        recipe = deserialize(f.readlines())

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
