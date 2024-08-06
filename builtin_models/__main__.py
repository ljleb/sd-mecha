import time
from builtin_models.script_runner import generate_model_configs


if __name__ == "__main__":
    start = time.time()
    generate_model_configs()
    print(f"total time taken: {time.time() - start}")
