import click
import pathlib
import torch
import safetensors.torch
import safetensors


@click.command()
@click.option("-i", "--input", "input_path", type=pathlib.Path)
@click.option("-o", "--output", "output_path", type=pathlib.Path)
def main(input_path: pathlib.Path, output_path: pathlib.Path):
    if not output_path.suffix == ".safetensors":
        print("can only convert to .safetensors")
        exit(1)

    print("loading model...")
    ckpt = torch.load(input_path)
    ckpt = ckpt.get("state_dict", ckpt)
    if "state_dict" in ckpt:
        del ckpt["state_dict"]

    print("saving...")
    safetensors.torch.save_file(ckpt, output_path)


if __name__ == "__main__":
    main()
