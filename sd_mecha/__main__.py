import click
import functools
import pathlib
import re
import torch
import traceback
import yaml
from sd_mecha import extensions
from sd_mecha.recipe_merger import RecipeMerger
from sd_mecha.recipe_nodes import ParameterResolverVisitor
from sd_mecha.recipe_serializer import deserialize, deserialize_path, serialize
from sd_mecha.user_error import UserError
from typing import Optional, Tuple, List


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
        if debug:
            click.echo("Debug mode active.", err=True)
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
@click.option("--models-directory", type=pathlib.Path, envvar="MECHA_MODELS_DIR", default=pathlib.Path(), help="Base directory for .safetensors checkpoints.")
@click.option("--output", "-o", type=pathlib.Path, default=None, help="Output file for the merge result.")
@click.option("--threads", "-j", default=1, help="Number of keys to merge in parallel.")
@click.option("--device", "-d", default="cpu", help="Device used for merging.")
@click.option("--dtype", "-p", type=click.Choice(list(DTYPE_MAPPING.keys())), default="fp32", help="Work precision.")
@click.option("--save-dtype", "-t", type=click.Choice(list(DTYPE_MAPPING.keys())), default="fp16", help="Save precision.")
@click.option("--debug", is_flag=True, help="Print the stacktrace when an error occurs.")
@except_fallback
def merge(
    recipe: pathlib.Path,
    models_directory: pathlib.Path,
    output: pathlib.Path,
    threads: int,
    device: str,
    dtype: str,
    save_dtype: str,
    debug: bool,
):
    if output is None:
        output = models_directory / "merge.safetensors"

    recipe = deserialize(recipe)
    scheduler = RecipeMerger(
        models_dir=models_directory,
        default_device=device,
        default_dtype=DTYPE_MAPPING[dtype],
    )
    scheduler.merge_and_save(
        recipe,
        output=output,
        save_dtype=DTYPE_MAPPING[save_dtype],
        threads=threads
    )


@click.command(help="Compose recipes together to form a larger recipe")
@click.argument("base_recipe", type=pathlib.Path)
@click.option("--models-directory", type=pathlib.Path, envvar="MECHA_MODELS_DIR", default=pathlib.Path(), help="Base directory for .safetensors checkpoints.")
@click.option("arguments", "--argument", "-p", type=(str, str), multiple=True, help="Recipe or model argument to pass to the base recipe.")
@click.option("--output", "-o", type=pathlib.Path, default=None, help="Output file of the composed recipe.")
@click.option("--debug", is_flag=True, help="Print the stacktrace when an error occurs.")
@except_fallback
def compose(
    base_recipe: pathlib.Path,
    models_directory: pathlib.Path,
    arguments: List[Tuple[str, str]],
    output: Optional[pathlib.Path],
    debug: bool,
):
    if output is None:
        output = pathlib.Path("merge.mecha")

    base_recipe = deserialize(base_recipe)
    arguments = {
        k: deserialize_path(v, models_directory)
        for k, v in arguments
    }
    composed_recipe = base_recipe.accept(ParameterResolverVisitor(arguments))
    composed_recipe = serialize(composed_recipe)
    with open(output, "w") as f:
        f.write(composed_recipe)


@click.command(help="Show the available blocks of model architectures")
@click.argument("model_arch", type=click.Choice(extensions.model_arch.get_all() + [""], case_sensitive=False), default="")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--debug", is_flag=True, help="Print the stacktrace when an error occurs.")
@except_fallback
def info(
    model_arch: str,
    verbose: bool,
    debug: bool,
):
    if not model_arch:
        print(yaml.dump({
            "Available architectures": [
                arch
                if not verbose else
                f"{arch} (registered at {extensions.model_arch.resolve(arch).location})"
                for arch in extensions.model_arch.get_all()
            ]
        }))
        return

    model_arch = extensions.model_arch.resolve(model_arch)
    print(yaml.dump({
        f'Component "{component}"': {
            "Blocks":
                sorted(list({
                    b.split("_", 3)[3]
                    for bs in model_arch.blocks.values()
                    for b in bs
                    if f"_{component}_block_" in b
                }), key=natural_sort_key)
                if not verbose else
                dict(sorted([
                    (b.split("_", 3)[3], sorted([
                        k for k in model_arch.blocks
                        if b in model_arch.blocks[k]
                    ], key=natural_sort_key))
                    for bs in model_arch.blocks.values()
                    for b in bs
                    if f"_{component}_block_" in b
                ], key=lambda t: natural_sort_key(t[0]))),
        }
        for component in model_arch.components
    }, sort_keys=False))


def natural_sort_key(s):
    return [
        int(text)
        if text.isdigit()
        else text.lower()
        for text in re.split('([0-9]+)', s)
    ]


main.add_command(merge)
main.add_command(compose)
main.add_command(info)


if __name__ == "__main__":
    main()
