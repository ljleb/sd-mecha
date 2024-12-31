import pathlib
from typing import List, Optional
from sd_mecha import extensions, recipe_nodes
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, RecipeVisitor


MECHA_FORMAT_VERSION = "0.1.0"


def deserialize_path(recipe: str | pathlib.Path, models_dir: Optional[str | pathlib.Path] = None) -> RecipeNode:
    if isinstance(recipe, str):
        recipe = pathlib.Path(recipe)
    if isinstance(models_dir, str):
        models_dir = pathlib.Path(models_dir)

    if models_dir is not None and not recipe.exists() and not recipe.is_absolute():
        recipe = models_dir / recipe

    if not recipe.exists():
        raise ValueError(f"unable to deserialize '{recipe}': no such file")

    if recipe.suffix == ".mecha":
        with open(recipe, "r") as recipe:
            return deserialize(recipe.read())
    else:
        raise ValueError(f"unable to deserialize '{recipe}': unknown extension")


def deserialize(recipe: List[str] | str) -> RecipeNode:
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
        positional_args, named_args = preprocess_args(args)
        if command == "dict":
            results.append(dict(*positional_args, **named_args))
        elif command == "model":
            results.append(ModelRecipeNode(*positional_args, **named_args))
        elif command == "merge":
            method, *positional_args = positional_args
            method = extensions.merge_method.resolve(method)
            results.append(method(*positional_args, **named_args))
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
            if arg == "null":
                return None
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


def serialize(recipe: RecipeNode) -> str:
    serializer = SerializerVisitor()
    recipe.accept(serializer)
    body = "\n".join(serializer.instructions)
    header = get_version_header(MECHA_FORMAT_VERSION)
    return f"{header}\n{body}"


def get_version_header(version: str):
    return f"version {version}"


class SerializerVisitor(RecipeVisitor):
    def __init__(self, instructions: Optional[List[str]] = None):
        self.instructions = instructions if instructions is not None else []

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> str:
        path = self.__serialize_value(node.path)
        config = self.__serialize_value(getattr(node.model_config, "identifier", None))
        line = f'model {path} {config}'
        return self.__add_instruction(line)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> str:
        args = [
            self.__serialize_value(v)
            for v in node.args
        ]
        kwargs = [
            f"{k}={self.__serialize_value(v)}"
            for k, v in node.kwargs.items()
        ]
        line = f'merge {self.__serialize_value(node.merge_method.get_identifier())} {" ".join(args)} {" ".join(kwargs)}'
        return self.__add_instruction(line)

    def __serialize_value(self, value) -> str:
        if value is None:
            return "null"
        if isinstance(value, (str, pathlib.Path)):
            value = str(value).replace("\\", "\\\\").replace('"', "\\\"")
            return f'"{value}"'
        if isinstance(value, dict):
            dict_line = "dict " + " ".join(f"{k}={self.__serialize_value(v)}" for k, v in value.items())
            return self.__add_instruction(dict_line)
        if isinstance(value, recipe_nodes.RecipeNode):
            return value.accept(self)
        if isinstance(value, (int, float)):
            return str(value)
        raise ValueError(value)

    def __add_instruction(self, instruction: str) -> str:
        try:
            return f"&{self.instructions.index(instruction)}"
        except ValueError:
            self.instructions.append(instruction)
            return f"&{len(self.instructions) - 1}"
