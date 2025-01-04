import dataclasses
import pathlib
from typing import Dict
import sd_mecha.streaming
from sd_mecha.extensions import model_config


@dataclasses.dataclass
class ModelConfig2:
    identifier: str
    keys: Dict[str, sd_mecha.streaming.TensorMetadata]


def main():
    for config in model_config.get_all_base():
        yaml_config = model_config.to_yaml(ModelConfig2(config.identifier, dict(sorted(list(config.compute_keys().items()), key=lambda t: t[0]))))
        with open(pathlib.Path(r"D:\src\sd-mecha\flat_configs") / f"{config.identifier}.yaml", "w") as f:
            f.write(yaml_config)


if __name__ == "__main__":
    main()
