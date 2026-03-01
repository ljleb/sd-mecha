from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from torch import Tensor


@merge_method(is_interface=True)
def exchange_ema(
    model: Parameter(StateDict[Tensor]),
) -> Return(Tensor):
    ...


def implement_exchange_ema(config, to_ema_key_fn):
    @merge_method(identifier=f"exchange_ema_'{config}'", implements=exchange_ema)
    def exchange_ema_impl(
        model: Parameter(StateDict[Tensor], model_config=config),
        **kwargs,
    ) -> Return(Tensor, model_config=config):
        input_keys = model.model_config.keys()
        target_key = kwargs["key"]
        ema_key = to_ema_key_fn(target_key)

        if ema_key in input_keys:
            return model[ema_key]

        for input_key in input_keys:
            if to_ema_key_fn(input_key) == target_key:
                return model[input_key]
        return model[target_key]


to_ema_key_fns = {
    "sd1-ldm": lambda k: f"model_ema.{k[len('model.'):].replace('.', '')}"
}


for config, to_ema_key_fn in to_ema_key_fns.items():
    implement_exchange_ema(config, to_ema_key_fn)
