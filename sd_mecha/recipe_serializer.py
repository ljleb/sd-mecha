from typing import List
from sd_mecha import extensions
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, LoraRecipeNode


def deserialize(recipe: List[str]) -> RecipeNode:
    results = []

    def tokenize(line):
        """Custom tokenizer to handle quoted strings and regular arguments."""
        tokens = []
        current_token = []
        inside_quotes = False
        for char in line:
            if char == '"':  # Toggle inside_quotes
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
        if current_token:  # Add last token if exists
            tokens.append(''.join(current_token))
        return tokens

    def get_arg_value(arg):
        if arg.startswith('&'):
            return results[int(arg[1:])]
        elif arg.startswith('"') and arg.endswith('"'):
            return arg[1:-1]
        elif '.' in arg or 'e' in arg.lower():
            return float(arg)
        else:
            return int(arg)

    def preprocess_args(args):
        positional_args = []
        named_args = {}
        for arg in args:
            if '=' in arg:
                key, value = arg.split('=')
                named_args[key] = get_arg_value(value)
            else:
                positional_args.append(get_arg_value(arg))
        return positional_args, named_args

    for line in recipe:
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
            method = extensions.methods_registry[method]
            results.append(method(*positional_args, **named_args))
        else:
            raise ValueError(f"Unknown command: {command}")

    return results[-1]


def serialize(recipe: RecipeNode) -> str:
    instructions = []
    recipe.serialize(instructions)
    return "\n".join(instructions)
