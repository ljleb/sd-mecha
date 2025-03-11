from sd_mecha import merge_method, Parameter, Return, StateDict
from torch import Tensor


@merge_method
def exchange_ema(
    model: Parameter(StateDict[Tensor]),
    **kwargs,
) -> Return(Tensor):
    input_keys = model.model_config.keys
    target_key = kwargs["key"]
    exchange_fn = ema_fns.get(model.model_config.identifier, lambda k: k)
    ema_key = exchange_fn(target_key)

    if ema_key in input_keys:
        return model[ema_key]
    else:
        for input_key in input_keys:
            if exchange_fn(input_key) == target_key:
                return model[input_key]
        return model[target_key]


ema_fns = {
    "sd1-ldm": lambda k: f"{SD1_EMA_PREFIX}.{k[len('model'):].replace('.', '')}"
}


SD1_EMA_PREFIX = "model_ema"
