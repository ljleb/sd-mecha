import torch
import sd_mecha
from sd_mecha.extensions import model_configs


def _register_aux(config):
    try:
        model_configs.register_aux(config)
    except ValueError:
        pass


config_without_handler = model_configs.ModelConfigImpl(
    "testmissing-withouthandler",
    {
        "component": model_configs.ModelComponent({
            "key": model_configs.KeyMetadata([2, 2], torch.float32),
        }),
    },
)
_register_aux(config_without_handler)


config_with_handler = model_configs.ModelConfigImpl(
    "testmissing-withhandler",
    {
        "component": model_configs.ModelComponent({
            "key": model_configs.KeyMetadata([2, 2], torch.float32),
        }),
    },
    missing_key_handler="tests.stubs.missing_key_handler.reconstruct_test_key",
)
_register_aux(config_with_handler)


def test_merge_without_missing_key_handler_still_skips_non_finite_key():
    recipe = sd_mecha.model(
        {"key": torch.full((2, 2), float("nan"), dtype=torch.float32)},
        config=config_without_handler,
    )

    output = sd_mecha.merge(
        recipe,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device="cpu",
        output_dtype=torch.float32,
    )

    assert "key" not in output


def test_merge_uses_missing_key_handler_for_non_finite_key():
    recipe = sd_mecha.model(
        {"key": torch.full((2, 2), float("nan"), dtype=torch.float32)},
        config=config_with_handler,
    )

    output = sd_mecha.merge(
        recipe,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device="cpu",
        output_dtype=torch.float64,
    )

    assert torch.equal(output["key"], torch.full((2, 2), 42, dtype=torch.float64))
