import logging
import pathlib
from .recipe_serializer import serialize
from .recipe_nodes import ModelRecipeNode, LiteralRecipeNode, RecipeNode
from .extensions.merge_methods import NonDictLiteralValue
from typing import Optional


def serialize_and_save(
    recipe: RecipeNode,
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


def model(path: str | pathlib.Path, model_config: Optional[str] = None) -> ModelRecipeNode:
    if isinstance(path, str):
        path = pathlib.Path(path)
    return ModelRecipeNode(path, model_config=model_config)


def literal(value: NonDictLiteralValue | dict, model_config: Optional[str] = None) -> LiteralRecipeNode:
    return LiteralRecipeNode(value, model_config=model_config)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
