import logging
import pathlib
from typing import Optional
from .recipe_serializer import serialize
from .extensions.merge_method import recipe, NonDictLiteralValue
from . import recipe_nodes


def serialize_and_save(
    recipe: recipe_nodes.RecipeNode,
    output_path: pathlib.Path | str,
):
    serialized = serialize(recipe)

    if not isinstance(output_path, pathlib.Path):
        output_path = pathlib.Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mecha")
    output_path = output_path.absolute()

    logging.info(f"Saving recipe to {output_path}")
    with open(output_path, "w") as f:
        f.write(serialized)


def model(path: str | pathlib.Path, model_config: Optional[str] = None) -> recipe_nodes.ModelRecipeNode:
    if isinstance(path, str):
        path = pathlib.Path(path)
    return recipe_nodes.ModelRecipeNode(path, model_config)


def literal(value: NonDictLiteralValue | dict, model_config: Optional[str] = None) -> recipe_nodes.LiteralRecipeNode:
    return recipe_nodes.LiteralRecipeNode(value, model_config)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
