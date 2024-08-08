import inspect
import torch


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
            kwargs = kwargs.copy()
            kwargs["device"] = "meta"
            return kwargs

        def to_meta_device(f, args, kwargs):
            args = [
                arg
                for arg in args
                if not isinstance(arg, (str, torch.device))
            ]
            if "device" in kwargs:
                del kwargs["device"]
            return f(*args, **kwargs)

        def patch_original_init(f, args, kwargs):
            try:
                return f(*args, **force_meta_device(kwargs))
            except (KeyError, ValueError, TypeError):
                return f(*args, **kwargs)

        for module_key, module_class in (torch.nn.__dict__ | torch.__dict__).items():
            if type(module_class) is not type or not issubclass(module_class, torch.nn.Module) or module_class is torch.nn.Module:
                continue
            spec = inspect.getfullargspec(module_class.__init__)
            if not spec.varkw and not any("device" in l for l in (spec.args, spec.kwonlyargs)):
                continue

            self.replace(
                module_class, "__init__",
                lambda *args, __original_init=module_class.__init__, **kwargs: patch_original_init(__original_init, args, kwargs)
            )

        torch_nn_Module_to = self.replace(torch.nn.Module, 'to', lambda *args, **kwargs: to_meta_device(torch_nn_Module_to, args, kwargs))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


class DisableInitialization(ReplaceHelper):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            return self.create_model_and_transforms(*args, pretrained=None, **kwargs)

        self.replace(torch.nn.init, 'kaiming_uniform_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_normal_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_uniform_', do_nothing)

        try:
            import open_clip
            self.create_model_and_transforms = self.replace(open_clip, 'create_model_and_transforms', create_model_and_transforms_without_pretrained)
        except ImportError:
            pass

        try:
            import transformers

            def create_model_from_config(*args, **kwargs):
                config = transformers.AutoConfig.from_pretrained(*args, **kwargs)
                return transformers.AutoModel.from_config(config)

            for module_name in transformers.__all__:
                try:
                    module_class = getattr(transformers, module_name)
                except (ImportError, RuntimeError):
                    continue

                if module_name == "AutoConfig" or not hasattr(module_class, "from_pretrained"):
                    print(f"NOT patching 'from_pretrained' of {module_name}")
                    continue

                self.replace(module_class, "from_pretrained", create_model_from_config)
                print(f"patching 'from_pretrained' of {module_name}")
        except ImportError:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
