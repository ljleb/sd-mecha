from sd_mecha import merge_method, Parameter, Return, StateDict
from torch import Tensor


@merge_method
def exchange_ema(
    model: Parameter(StateDict[Tensor]),
    **kwargs,
) -> Return(Tensor):
    input_keys = model.model_config.keys()
    target_key = kwargs["key"]
    to_ema_key_fn = to_ema_key_fns.get(model.model_config.identifier, lambda k: k)
    ema_key = to_ema_key_fn(target_key)

    if ema_key in input_keys:
        return model[ema_key]
    else:
        for input_key in input_keys:
            if to_ema_key_fn(input_key) == target_key:
                return model[input_key]
        return model[target_key]


to_ema_key_fns = {
    "sd1-ldm": lambda k: f"model_ema.{k[len('model.'):].replace('.', '')}"
}
