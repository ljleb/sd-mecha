import click
import pathlib
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
    ckpt = {}
    with safetensors.safe_open(input_path, "pt") as f:
        for k in f.keys():
            ckpt[k] = f.get_tensor(k)

    print(len(ckpt))
    with safetensors.safe_open(r"E:\sd\models\Stable-diffusion\pure\sdxl_base.safetensors", "pt") as f:
        for k in f.keys():
            if k not in ckpt:
                ckpt[k] = f.get_tensor(k)

    print(len(ckpt))
    print("saving...")


if __name__ == "__main__":
    main()
