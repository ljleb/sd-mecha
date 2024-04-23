import pathlib
from typing import List, Optional
from sd_mecha import extensions, recipe_nodes
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, MergeSpace, ParameterRecipeNode, RecipeVisitor


def deserialize_path(recipe: str | pathlib.Path, models_dir: Optional[str | pathlib.Path] = None) -> RecipeNode:
    if not isinstance(recipe, pathlib.Path):
        recipe = pathlib.Path(recipe)
    if recipe.exists():
        if recipe.suffix == ".safetensors":
            return recipe_nodes.ModelRecipeNode(recipe)
        elif recipe.suffix == ".mecha":
            return deserialize(recipe)
        else:
            raise ValueError(f"unable to deserialize '{recipe}': unknown extension")

    if models_dir is not None and not recipe.is_absolute():
        recipe = models_dir / recipe
    if not recipe.suffix:
        recipe = recipe.with_suffix(".safetensors")
    elif recipe.suffix != ".safetensors":
        raise ValueError(f"invalid path to safetensors or recipe: {recipe}")
    return recipe_nodes.ModelRecipeNode(recipe)


def deserialize(recipe: List[str] | str | pathlib.Path) -> RecipeNode:
    if not isinstance(recipe, list):
        with open(recipe, "r") as recipe:
            recipe = recipe.readlines()

    results = []

    def parse(line):
        command, *args = tokenize(line.strip())
        positional_args, named_args = preprocess_args(args)
        if command == "dict":
            results.append(dict(*positional_args, **named_args))
        elif command == "model":
            results.append(ModelRecipeNode(*positional_args, **named_args))
        elif command == "parameter":
            merge_space = MergeSpace[positional_args[1]]
            results.append(ParameterRecipeNode(positional_args[0], merge_space, **named_args))
        elif command == "merge":
            method, *positional_args = positional_args
            method = extensions.merge_method.resolve(method)
            results.append(method.create_recipe(*positional_args, **named_args))
        else:
            raise ValueError(f"unknown command: {command}")

    def preprocess_args(args):
        positional_args = []
        named_args = {}
        for arg_index, arg in enumerate(args):
            if '=' in arg:
                # note: this is wrong if "=" is inside quotes
                # however, quoted kwarg values (aka string hypers) will raise an exception in the constructor of recipes
                # so I'm not gonna bother fixing this with the tokenizer until it is actually useful to do so
                key, value = arg.split('=')
                named_args[key] = get_arg_value(value, arg_index)
            else:
                positional_args.append(get_arg_value(arg, arg_index))
        return positional_args, named_args

    def get_arg_value(arg, arg_index):
        try:
            if arg.startswith('&'):
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
        inside_quotes = False
        for char in line:
            if char == '"':
                inside_quotes = not inside_quotes
                if not inside_quotes:  # End of quoted string
                    tokens.append(f'"{"".join(current_token)}"')
                    current_token = []
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
    return "\n".join(serializer.instructions)


class SerializerVisitor(RecipeVisitor):
    def __init__(self, instructions: Optional[List[str]] = None):
        self.instructions = instructions if instructions is not None else []

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> int:
        line = f'model "{node.path}" "{node.model_arch.identifier}" "{node.model_type.identifier}"'
        return self.__add_instruction(line)

    def visit_parameter(self, node: recipe_nodes.ParameterRecipeNode) -> int:
        merge_space = str(MergeSpace.BASE).split(".")[1]
        model_arch = f' "{node.model_arch.identifier}"' if node.model_arch else ""
        line = f'parameter "{node.name}" "{merge_space}"' + model_arch
        return self.__add_instruction(line)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> int:
        models = [
            f" &{model.accept(self)}"
            for model in node.models
        ]

        hypers = []
        for hyper_k, hyper_v in node.hypers.items():
            if isinstance(hyper_v, dict):
                dict_line = "dict" + "".join(f" {k}={v}" for k, v in hyper_v.items())
                hyper_v = f"&{self.__add_instruction(dict_line)}"
            hypers.append(f" {hyper_k}={hyper_v}")

        line = f'merge "{node.merge_method.get_name()}"{"".join(models)}{"".join(hypers)}'
        return self.__add_instruction(line)

    def __add_instruction(self, instruction: str) -> int:
        try:
            return self.instructions.index(instruction)
        except ValueError:
            self.instructions.append(instruction)
            return len(self.instructions) - 1
