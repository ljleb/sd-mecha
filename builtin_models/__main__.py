import time
from builtin_models.script_runner import generate_model_configs
from builtin_models.script_venvs import create_venvs


if __name__ == "__main__":
    start = time.time()
    create_venvs()
    generate_model_configs()
    print(f"total time taken: {time.time() - start}")
