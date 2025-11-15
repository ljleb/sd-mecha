import logging
import pathlib
from typing import List, Optional, Hashable
from .extensions import merge_methods
from .recipe_nodes import RecipeNode, ModelRecipeNode, RecipeVisitor, LiteralRecipeNode, MergeRecipeNode


MECHA_FORMAT_VERSION = "0.1.0"


def deserialize_path(recipe: pathlib.Path) -> RecipeNode:
    if not recipe.exists():
        raise ValueError(f"unable to deserialize '{recipe}': no such file")

    if recipe.suffix == ".mecha":
        with open(recipe, "r") as recipe_file:
            return deserialize(recipe_file.read())
    else:
        raise ValueError(f"unable to deserialize '{recipe}': unknown extension")


def deserialize(recipe: str | List[str]) -> RecipeNode:
    """
    Recreate a recipe graph from its serialized `.mecha` format.

    Args:
        recipe (str or List[str]):
            The textual representation (as a string or list of lines) of the recipe.

    Returns:
        A `RecipeNode` that can be further merged or manipulated.
    """
    if not isinstance(recipe, list):
        recipe = recipe.split("\n")

    if not recipe[0].startswith("version"):
        raise RuntimeError("bad format: expected version at line 1")

    actual_version, recipe = recipe[0], recipe[1:]
    expected_version = get_version_header(MECHA_FORMAT_VERSION)
    if actual_version != expected_version:
        raise RuntimeError(f"bad recipe version: got {actual_version}, expected {expected_version}")

    results = []

    def parse(line):
        line = line.strip()
        if line.startswith("#"):
            return

        command, *args = tokenize(line)
        positional_args, keyword_args = preprocess_args(args)
        if command == "dict":
            results.append(dict(*positional_args, **keyword_args))
        elif command == "literal":
            results.append(LiteralRecipeNode(*positional_args, **keyword_args))
        elif command == "model":
            path = pathlib.Path(positional_args[0])
            results.append(ModelRecipeNode(path, *positional_args[1:], **keyword_args))
        elif command == "merge":
            method, *positional_args = positional_args
            method = merge_methods.resolve(method)
            results.append(method(*positional_args, **keyword_args))
        else:
            raise ValueError(f"unknown command: {command}")

    def preprocess_args(args):
        positional_args = []
        named_args = {}
        for arg_index, arg in enumerate(args):
            if '=' in arg:
                key, value = arg.split('=', maxsplit=1)
                named_args[key] = get_arg_value(value, arg_index)
            else:
                positional_args.append(get_arg_value(arg, arg_index))
        return positional_args, named_args

    def get_arg_value(arg, arg_index):
        try:
            if arg in CONSTANTS:
                return CONSTANTS[arg]
            elif arg.startswith('&'):
                ref_index = int(arg[1:])
                if ref_index < 0 or ref_index >= len(results):
                    raise ValueError(f"reference {arg} out of bounds")
                return results[ref_index]
            elif arg.startswith('"') and arg.endswith('"'):
                return arg[1:-1]
            elif '.' in arg or 'e' in arg.lower():
                return float(arg)
            else:
                return int(arg)
        except ValueError as e:
            raise ValueError(f"argument {arg_index}: {str(e)}")

    def tokenize(line):
        tokens = []
        current_token = []
        quote_prefix = []
        inside_quotes = False
        is_escape = False
        for char in line:
            if is_escape:
                is_escape = False
            elif char == "\\":
                is_escape = True
                continue
            elif char == '"':
                inside_quotes = not inside_quotes
                if inside_quotes:  # Begin of quoted string
                    quote_prefix = current_token
                    current_token = []
                else:  # End of quoted string
                    tokens.append(f'{"".join(quote_prefix)}"{"".join(current_token)}"')
                    current_token = []
                    quote_prefix = []
                continue
            elif char == ' ' and not inside_quotes:
                if current_token:  # End of a token
                    tokens.append(''.join(current_token))
                    current_token = []
                continue
            current_token.append(char)
        if inside_quotes:  # Handle mismatched quotes
            raise ValueError(f"mismatched quotes in input")
        if current_token:  # Add last token if exists
            tokens.append(''.join(current_token))
        return tokens

    for line_num, line in enumerate(recipe, 1):
        try:
            parse(line)
        except ValueError as e:
            raise ValueError(f"line {line_num}: {e}.\n    {line}")

    return results[-1]


def serialize(recipe: RecipeNode, *, output: Optional[pathlib.Path | str] = None) -> str:
    """
    Convert a recipe graph into a string that captures its merge instructions.

    This is the first step of persisting a recipe to disk in `.mecha` format.

    Args:
        recipe:
            A `RecipeNode` describing the merge.
        output:
            Path to the output file to save.

    Returns:
        A string representation of the recipe, suitable for writing to a .mecha file.
    """
    serializer = SerializerVisitor()
    recipe.accept(serializer)
    version_header = get_version_header(MECHA_FORMAT_VERSION)
    serialized = "\n".join([version_header] + serializer.instructions)

    if isinstance(output, str):
        output = pathlib.Path(output)
    if output is not None:
        output = output.absolute()
        logging.info(f"Saving recipe to {output}")
        output.write_text(serialized)

    return serialized


def get_version_header(version: str):
    return f"version {version}"


class SerializerVisitor(RecipeVisitor):
    def __init__(self, instructions: Optional[List[str]] = None):
        self.instructions = instructions if instructions is not None else []

    def visit_literal(self, node: LiteralRecipeNode):
        value = self.__serialize_value(node.value)
        if node.model_config is None:
            return value
        else:
            config = self.__serialize_value(node.model_config.identifier)
            merge_space = self.__serialize_value(node.merge_space.identifier)
            line = f"literal {value} model_config={config} merge_space={merge_space}"
            return self.__add_instruction(line)

    def visit_model(self, node: ModelRecipeNode) -> str:
        path = self.__serialize_value(str(node.path))
        config = self.__serialize_value(getattr(node.model_config, "identifier", None))
        merge_space = self.__serialize_value(node.merge_space.identifier)
        line = f"model {path} model_config={config} merge_space={merge_space}"
        return self.__add_instruction(line)

    def visit_merge(self, node: MergeRecipeNode) -> str:
        identifier = self.__serialize_value(node.merge_method.get_identifier())
        parts = ["merge", identifier] + [
            self.__serialize_value(v)
            for v in node.args
        ] + [
            f"{k}={self.__serialize_value(v)}"
            for k, v in node.kwargs.items()
        ]
        line = " ".join(parts)
        return self.__add_instruction(line)

    def __serialize_value(self, value) -> str:
        if isinstance(value, str):
            value = value.replace("\\", "\\\\").replace('"', "\\\"")
            return f'"{value}"'
        if isinstance(value, dict):
            dict_line = "dict " + " ".join(f"{k}={self.__serialize_value(v)}" for k, v in value.items())
            return self.__add_instruction(dict_line)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        # int or float needs to be handled before this (1.0 == True)
        if isinstance(value, Hashable) and value in REVERSE_CONSTANTS:
            return REVERSE_CONSTANTS[value]
        if isinstance(value, RecipeNode):
            return value.accept(self)
        raise TypeError(f"type {type(value)} cannot be serialized: {value}")

    def __add_instruction(self, instruction: str) -> str:
        try:
            return f"&{self.instructions.index(instruction)}"
        except ValueError:
            self.instructions.append(instruction)
            return f"&{len(self.instructions) - 1}"


CONSTANTS = {
    "null": None,
    "true": True,
    "false": False,
}
REVERSE_CONSTANTS = {
    v: k for k, v in CONSTANTS.items()
}
