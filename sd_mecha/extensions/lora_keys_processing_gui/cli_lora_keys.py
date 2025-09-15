import safetensors
import os
import sys
import argparse
import glob

# IMPORTANT:
# Requires torch installation in the environment

# Process specific files
#python SaveLoRAKeys.py mymodel.safetensors

# Process everything in a folder
#python SaveLoRAKeys.py /path/to/lora/models

# Mix files and folders
#python SaveLoRAKeys.py model1.safetensors model2.safetensors ./my_loras/

# To print output to stdout instead of saving to files, add the --stdout argument:
# python SaveLoRAKeys.py mymodel.safetensors --stdout

def process_file(path: str, to_stdout: bool = False):
    if to_stdout:
        try:
            with safetensors.safe_open(path, "pt") as f:
                print(f"--- Keys for {os.path.basename(path)} ---")
                for k in f.keys():
                    shape = list(f.get_tensor(k).shape)
                    line = f"{k} {shape}"
                    print(line)
            return None
        except Exception as e:
            return f"Failed to process {path}: {e}"
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")

        try:
            with safetensors.safe_open(path, "pt") as f, open(out_path, "w") as out_file:
                for k in f.keys():
                    shape = list(f.get_tensor(k).shape)
                    line = f"{k} {shape}\n"
                    out_file.write(line)
            return None
        except Exception as e:
            return f"Failed to process {path}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Process LoRA safetensors files")
    parser.add_argument('paths', nargs='+', help='File paths or directory paths containing .safetensors files')
    parser.add_argument('--stdout', action='store_true', help='Print output to stdout instead of saving to files')
    args = parser.parse_args()

    files_to_process = []
    for path in args.paths:
        if os.path.isfile(path):
            if path.endswith('.safetensors'):
                files_to_process.append(path)
        elif os.path.isdir(path):
            # Find all .safetensors files in directory recursively
            pattern = os.path.join(path, '**', '*.safetensors')
            files_to_process.extend(glob.glob(pattern, recursive=True))

    if not files_to_process:
        print("No .safetensors files found in the provided paths.")
        sys.exit(1)

    print(f"Found {len(files_to_process)} .safetensors files to process:")
    for f in files_to_process:
        print(f"  {f}")

    errors = []
    successful = 0

    for file_path in files_to_process:
        error = process_file(file_path, args.stdout)
        if error:
            errors.append(error)
        else:
            successful += 1

    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for error in errors:
            print(f"    {error}")

if __name__ == "__main__":
    main()
