import pathlib
from collections import OrderedDict
from sd_mecha.extensions import model_configs
from sd_mecha.streaming import InSafetensorsDict


cfg_id = "sdxl-diffusers"
path_to_model = pathlib.Path(r"E:\sd\models\unet\sdxl-unet.safetensors")
path_to_config = pathlib.Path(".") / f"{cfg_id}.yaml"


def main():
    header = OrderedDict(InSafetensorsDict(path_to_model, 0).metadata())
    config = model_configs.ModelConfigImpl(cfg_id, {
        "diffuser": model_configs.ModelComponent(header)
    })
    model_config_str = model_configs.to_yaml(config)
    with open(path_to_config, "w") as f:
        f.write(model_config_str)


if __name__ == "__main__":
    main()
