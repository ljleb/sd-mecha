from typing import List
from sd_mecha import extensions
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, LoraRecipeNode


def deserialize(recipe: List[str]) -> RecipeNode:
    results = []

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
            raise ValueError(f"Mismatched quotes in input.")
        if current_token:  # Add last token if exists
            tokens.append(''.join(current_token))
        return tokens

    def get_arg_value(arg, arg_index):
        try:
            if arg.startswith('&'):
                ref_index = int(arg[1:])
                if ref_index < 0 or ref_index >= len(results):
                    raise ValueError(f"Reference {arg} out of bounds.")
                return results[ref_index]
            elif arg.startswith('"') and arg.endswith('"'):
                return arg[1:-1]
            elif '.' in arg or 'e' in arg.lower():
                return float(arg)
            else:
                return int(arg)
        except ValueError as e:
            raise ValueError(f"Argument {arg_index}: {str(e)}")

    def preprocess_args(args):
        positional_args = []
        named_args = {}
        for arg_index, arg in enumerate(args):
            if '=' in arg:
                key, value = arg.split('=')
                named_args[key] = get_arg_value(value, arg_index)
            else:
                positional_args.append(get_arg_value(arg, arg_index))
        return positional_args, named_args

    for line_num, line in enumerate(recipe, 1):
        try:
            command, *args = tokenize(line.strip())
            positional_args, named_args = preprocess_args(args)
            if command == "dict":
                results.append(dict(*positional_args, **named_args))
            elif command == "model":
                results.append(ModelRecipeNode(*positional_args, **named_args))
            elif command == "lora":
                results.append(LoraRecipeNode(*positional_args, **named_args))
            elif command == "call":
                method, *positional_args = positional_args
                try:
                    method = extensions.methods_registry[method]
                except KeyError as e:
                    raise ValueError(f"Unknown merge method: {e}")
                results.append(method(*positional_args, **named_args))
            else:
                raise ValueError(f"Unknown command: {command}")
        except ValueError as e:
            raise ValueError(f"Line {line_num}: {e}:\n    {line}")

    return results[-1]


def serialize(recipe: RecipeNode) -> str:
    instructions = []
    recipe.serialize(instructions)
    return "\n".join(instructions)
