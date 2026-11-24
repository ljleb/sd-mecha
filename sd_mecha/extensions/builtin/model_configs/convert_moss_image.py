from dataclasses import replace

from torch import Tensor

from sd_mecha.extensions import model_configs
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict

from .convert_n_to_1 import convert_1_to_n, convert_n_to_1, pack_key_requirements


openmoss_config = model_configs.resolve("moss_image-openmoss")


def _build_comfyui_config():
    components = {}
    source_keys = {}
    for component_name, component in openmoss_config.components().items():
        openmoss_keys = component.keys()
        comfyui_keys = {}
        for key, metadata in openmoss_keys.items():
            if component_name == "diffuser" and key.endswith((".proj.k.weight", ".proj.v.weight")):
                continue

            sources = (key,)
            if component_name == "diffuser" and key.endswith(".proj.q.weight"):
                stem = key.removesuffix(".q.weight")
                sources = tuple(f"{stem}.{part}.weight" for part in ("q", "k", "v"))
                key = f"{stem}.qkv.weight"
                metadata = replace(
                    metadata,
                    shape=(sum(openmoss_keys[source].shape[0] for source in sources), *metadata.shape[1:]),
                    aliases=tuple(alias.removesuffix(".q.weight") + ".qkv.weight" for alias in metadata.aliases),
                )

            comfyui_keys[key] = metadata
            source_keys[key] = sources
        components[component_name] = model_configs.ModelComponent(comfyui_keys)

    return model_configs.ModelConfigImpl("moss_image-comfyui", components), source_keys


# The ComfyUI checkpoint packs the OpenMOSS projections along the output dimension in Q, K, V order.
comfyui_config, _openmoss_keys_by_comfyui_key = _build_comfyui_config()
model_configs.register(comfyui_config)


@merge_method(is_conversion=True, reuse_outputs=False)
class convert_moss_image_openmoss_to_comfyui:
    @staticmethod
    def map_keys(b):
        for key, sources in _openmoss_keys_by_comfyui_key.items():
            b[key] = pack_key_requirements(b, sources) @ sources

    def __call__(
        self,
        state_dict: Parameter(StateDict[Tensor], model_config=openmoss_config),
        **kwargs,
    ) -> Return(Tensor, model_config=comfyui_config):
        relation = kwargs["key_relation"]
        return convert_n_to_1(state_dict, relation.inputs["state_dict"], source_keys=relation.meta)


@merge_method(is_conversion=True, reuse_outputs=False)
class convert_moss_image_comfyui_to_openmoss:
    @staticmethod
    def map_keys(b):
        for key, sources in _openmoss_keys_by_comfyui_key.items():
            b[sources] = b.keys[key]

    def __call__(
        self,
        state_dict: Parameter(StateDict[Tensor], model_config=comfyui_config),
        **kwargs,
    ) -> Return(StateDict[Tensor], model_config=openmoss_config):
        relation = kwargs["key_relation"]
        (key,) = relation.inputs["state_dict"]
        return convert_1_to_n(state_dict[key], relation.outputs)
