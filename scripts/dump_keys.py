"""
Extract tensor keys + shapes from .safetensors LoRA files.

Defaults to writing useful output to STDOUT. If --out-file or --out-dir is
provided, writes outputs there instead. Status/info/errors always go to STDERR.

Examples
--------
# Process a single file → STDOUT
$ python SaveLoRAKeys.py mymodel.safetensors

# Process everything in a folder (non‑recursive by default) → STDOUT
$ python SaveLoRAKeys.py /path/to/lora/models

# Mix files and folders → STDOUT
$ python SaveLoRAKeys.py model1.safetensors model2.safetensors ./my_loras/

# Write outputs into a single file (combined, with separators)
$ python SaveLoRAKeys.py ./my_loras --out-file all_keys.txt

# Write one .txt per input into a directory
$ python SaveLoRAKeys.py ./my_loras --out-dir ./keys

# Recurse into subdirectories
$ python SaveLoRAKeys.py ./my_loras -r
"""
import argparse
import sys
import safetensors
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, TextIO, Tuple


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.out_dir and args.out_file:
        print("Specify only one of --out-dir or --out-file, not both.", file=sys.stderr)
        return 2

    inputs = [Path(x) for x in args.paths]
    files = list_safetensors(inputs, recursive=args.recursive)
    if not files:
        print("No .safetensors files found in the provided paths.", file=sys.stderr)
        return 1

    print_file_list(files)

    try:
        if args.out_file:
            successes, errors = run_out_file(files, args.out_file, args.separator)
        elif args.out_dir:
            successes, errors = run_out_dir(files, args.out_dir, inputs)
        else:
            successes, errors = bundle_to_stream(files, sys.stdout, args.separator)
    except Exception as e:
        successes, errors = 0, [f"Unexpected error: {e}"]

    return print_summary(successes, errors)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract tensor keys and shapes from .safetensors (LoRA) files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("paths", nargs="+", help="Files or directories containing .safetensors")
    p.add_argument("--out-dir", type=Path, default=None, help="Directory to write one .txt per input (ignored if --out-file is used)")
    p.add_argument("--out-file", type=Path, default=None, help="Write all outputs into a single file (combined with separators)")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories when a directory is given")
    p.add_argument("--separator", default="---", help="Separator printed between files (for STDOUT and --out-file modes)")
    return p.parse_args(argv)


def list_safetensors(paths: Sequence[Path], recursive: bool = False) -> List[Path]:
    files = []
    for p in paths:
        if p.is_file() and p.suffix == ".safetensors":
            files.append(p)
        elif p.is_dir():
            globber = p.rglob if recursive else p.glob
            files.extend(sorted(globber("*.safetensors")))

    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique


def print_file_list(files: Sequence[Path]) -> None:
    print(f"Found {len(files)} .safetensors files to process:", file=sys.stderr)
    for f in files:
        print(f"  {f}", file=sys.stderr)


def run_out_file(files: Sequence[Path], out_file: Path, separator: str) -> Tuple[int, List[str]]:
    errors = []
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as outfh:
            successes, errs = bundle_to_stream(files, outfh, separator)
        print(f"Wrote {out_file}", file=sys.stderr)
        errors.extend(errs)
        return successes, errors
    except Exception as e:
        errors.append(f"Failed to open/write to {out_file}: {e}")
        return 0, errors


def run_out_dir(files: Sequence[Path], out_dir: Path, inputs: Sequence[Path]) -> Tuple[int, List[str]]:
    successes = 0
    errors = []
    for fpath in files:
        root = find_input_root(fpath, inputs)
        rel = fpath.relative_to(root) if root else Path(fpath.name)
        out_path = (out_dir / rel).with_suffix(".txt")
        try:
            lines = extract_keys_and_shapes(fpath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                write_lines(lines, fh)
            print(f"Wrote {out_path}", file=sys.stderr)
            successes += 1
        except Exception as e:
            errors.append(f"Failed to process {fpath}: {e}")
    return successes, errors


def bundle_to_stream(files: Sequence[Path], stream: TextIO, separator: str) -> Tuple[int, List[str]]:
    successes = 0
    errors = []
    for i, fpath in enumerate(files):
        file_separator = separator if len(files) > 1 and i < len(files) - 1 else None
        err = process_to_stream(fpath, stream, file_separator)
        if err:
            errors.append(err)
        else:
            successes += 1
    return successes, errors


def print_summary(successes: int, errors: List[str]) -> int:
    print("Processing complete:", file=sys.stderr)
    print(f"  Successful: {successes}", file=sys.stderr)
    if errors:
        print(f"  Errors: {len(errors)}", file=sys.stderr)
        for err in errors:
            print(f"    {err}", file=sys.stderr)
    return 0 if not errors else 1


def find_input_root(file: Path, inputs: Sequence[Path]) -> Optional[Path]:
    for r in inputs:
        if r.is_dir():
            try:
                file.relative_to(r)
                return r
            except ValueError:
                pass
    return None


def process_to_stream(fpath: Path, stream: TextIO, separator: Optional[str]) -> Optional[str]:
    try:
        lines = extract_keys_and_shapes(fpath)
        if separator is not None:
            stream.write(f"# {fpath}\n")
        write_lines(lines, stream)
        if separator is not None:
            stream.write(separator + "\n")
        return None
    except Exception as e:
        return f"Failed to process {fpath}: {e}"


def extract_keys_and_shapes(file_path: Path) -> List[str]:
    lines = []
    with safetensors.safe_open(str(file_path), "pt") as f:
        for k in f.keys():
            shape = list(f.get_tensor(k).shape)
            lines.append(f"{k} {shape}")
    return lines


def write_lines(lines: Iterable[str], stream: TextIO) -> None:
    for line in lines:
        if not line.endswith("\n"):
            line += "\n"
        stream.write(line)


if __name__ == "__main__":
    raise SystemExit(main())
