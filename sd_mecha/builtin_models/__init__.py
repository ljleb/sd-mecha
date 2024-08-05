import inspect
import logging
import omegaconf
import pathlib
import sys
import torch
from contextlib import contextmanager
from sd_mecha.extensions import model_impl
from typing import Iterable


@contextmanager
def extra_path(*paths):
    original_sys_path = sys.path.copy()
    sys.path[:0] = [str(path) for path in paths]
    yield
    sys.path[:] = original_sys_path


module_dir = pathlib.Path(__file__).parent
repositories_dir = module_dir / "repositories"


def register_builtin_models():
    for callback in (
        register_sd1_ldm_base,
    ):
        with extra_path(
            repositories_dir / "stability-ai-stable-diffusion",
            repositories_dir / "stability-ai-generative-models",
        ):
            try:
                callback()
            except ImportError as e:
                logging.error(e)


def register_sd1_ldm_base():
    import ldm.modules.encoders.modules as ldm_encoder_modules
    from ldm.util import instantiate_from_config
    config = str(module_dir / "configs/v1-inference.yaml")
    config = omegaconf.OmegaConf.load(config).model

    with DisableInitialization(ldm_encoder_modules), MetaTensorMode():
        model = instantiate_from_config(config)
        model_impl.register_model_autodetect(
            identifier="sd1-ldm-base",
            model=model,
            components=(
                model_impl.ModelComponent("unet", unet := model.model.diffusion_model, {
                    **list_blocks("in", unet.input_blocks.children()),
                    "mid": unet.middle_block,
                    **list_blocks("out", unet.output_blocks.children()),
                }),
                model_impl.ModelComponent("txt", txt := model.cond_stage_model.transformer.text_model, {
                    **list_blocks("in", txt.encoder.layers.children()),
                }),
            ),
            modules_to_ignore=model.first_stage_model,
        )


def list_blocks(block_id_prefix: str, modules: Iterable[torch.nn.Module]):
    return {
        f"{block_id_prefix}{i}": module
        for i, module in enumerate(modules)
    }


# src: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/modules/sd_disable_initialization.py#L9
class ReplaceHelper:
    def __init__(self):
        self.replaced = []

    def replace(self, obj, field, func):
        original = getattr(obj, field, None)
        if original is None:
            return None

        self.replaced.append((obj, field, original))
        setattr(obj, field, func)

        return original

    def restore(self):
        for obj, field, original in self.replaced:
            setattr(obj, field, original)

        self.replaced.clear()


class MetaTensorMode(ReplaceHelper):
    def __enter__(self):
        def force_meta_device(kwargs):
            kwargs["device"] = "meta"
            return kwargs

        def patch_original_init(f, args, kwargs):
            try:
                return f(*args, **force_meta_device(kwargs))
            except:
                return f(*args, **kwargs)

        for module_key, module_class in torch.nn.__dict__.items():
            if type(module_class) is not type or not issubclass(module_class, torch.nn.Module) or module_class is torch.nn.Module:
                continue
            spec = inspect.getfullargspec(module_class.__init__)
            if not spec.varkw and not any("device" in l for l in (spec.args, spec.kwonlyargs)):
                continue

            self.replace(
                module_class, "__init__",
                lambda *args, __original_init=module_class.__init__, **kwargs: patch_original_init(__original_init, args, kwargs)
            )

        self.replace(torch.nn.Module, 'to', lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


class DisableInitialization(ReplaceHelper):
    def __init__(self, ldm_encoder_modules=None):
        super().__init__()
        self.ldm_encoder_modules = ldm_encoder_modules

    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            return self.create_model_and_transforms(*args, pretrained=None, **kwargs)

        def CLIPTextModel_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
            res = self.CLIPTextModel_from_pretrained(None, *model_args, config=pretrained_model_name_or_path, state_dict={}, **kwargs)
            res.name_or_path = pretrained_model_name_or_path
            return res

        def transformers_modeling_utils_load_pretrained_model(*args, **kwargs):
            args = args[0:3] + ('/', ) + args[4:]  # resolved_archive_file; must set it to something to prevent what seems to be a bug
            return self.transformers_modeling_utils_load_pretrained_model(*args, **kwargs)

        def transformers_utils_hub_get_file_from_cache(original, url, *args, **kwargs):

            # this file is always 404, prevent making request
            if url is None or url == 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/added_tokens.json' or url == 'openai/clip-vit-large-patch14' and args[0] == 'added_tokens.json':
                return None

            try:
                res = original(url, *args, local_files_only=True, **kwargs)
                if res is None:
                    res = original(url, *args, local_files_only=False, **kwargs)
                return res
            except Exception:
                return original(url, *args, local_files_only=False, **kwargs)

        def transformers_tokenization_utils_base_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_tokenization_utils_base_cached_file, url, *args, **kwargs)

        def transformers_configuration_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_configuration_utils_cached_file, url, *args, **kwargs)

        def transformers_modeling_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_modeling_utils_cached_file, url, *args, **kwargs)

        def transformers_utils_hub_get_from_cache(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_utils_hub_get_from_cache, url, *args, **kwargs)

        self.replace(torch.nn.init, 'kaiming_uniform_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_normal_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_uniform_', do_nothing)

        import transformers
        import open_clip

        self.create_model_and_transforms = self.replace(open_clip, 'create_model_and_transforms', create_model_and_transforms_without_pretrained)
        if self.ldm_encoder_modules is not None:
            self.CLIPTextModel_from_pretrained = self.replace(self.ldm_encoder_modules.CLIPTextModel, 'from_pretrained', CLIPTextModel_from_pretrained)
        self.transformers_modeling_utils_load_pretrained_model = self.replace(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model', transformers_modeling_utils_load_pretrained_model)
        self.transformers_tokenization_utils_base_cached_file = self.replace(transformers.tokenization_utils_base, 'cached_file', transformers_tokenization_utils_base_cached_file)
        self.transformers_configuration_utils_cached_file = self.replace(transformers.configuration_utils, 'cached_file', transformers_configuration_utils_cached_file)
        self.transformers_modeling_utils_cached_file = self.replace(transformers.modeling_utils, 'cached_file', transformers_modeling_utils_cached_file)
        self.transformers_utils_hub_get_from_cache = self.replace(transformers.utils.hub, 'get_from_cache', transformers_utils_hub_get_from_cache)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


register_builtin_models()
