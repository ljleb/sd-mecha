import safetensors
import sys


if __name__ == "__main__":
    with safetensors.safe_open(sys.argv[1], "pt") as f:
        for k in f.keys():
            print(k)
