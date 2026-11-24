import warnings
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

import pytest
import torch
from lycoris.modules.boft import ButterflyOFTModule
from safetensors.torch import load_file, save_file

import sd_mecha
from sd_mecha import lokr_dims, lokr_dims_lycoris, truncate_kronecker
from sd_mecha.extensions import merge_methods
from sd_mecha.extensions.builtin.lycoris import _common
from sd_mecha.extensions.builtin.lycoris.boft import _cayley_blocks
from sd_mecha.graph_finalization import PropagatableKeyVisitor
from sd_mecha.keys_map import RealizedKeyRelation


CONFIG = "sd1-kohya"
WEIGHT = "vae.post_quant_conv.weight"
BIAS = "vae.post_quant_conv.bias"


def merge(recipe, space="weight"):
    output = {}
    sd_mecha.merge(recipe, output=output, output_dtype=torch.float32, strict_merge_space=space, threads=0)
    return output


@pytest.mark.parametrize("error", [FileNotFoundError("missing backend dependency"), RuntimeError("Failed to find C compiler.")])
def test_missing_lycoris_backend_dependency_only_affects_its_device(monkeypatch, error):
    monkeypatch.setattr(_common, "_unavailable_acceleration", set())
    first = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float32, is_cuda=True)
    second = SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.float32, is_cuda=True)
    calls = []

    def rebuild(weight, *, backend):
        calls.append((weight.device, backend))
        if weight is first and backend is None:
            raise error
        return weight

    with pytest.warns(UserWarning, match="could not start its accelerated backend on cuda:0"):
        assert _common.run_weight_function(rebuild, first) is first
    assert _common.run_weight_function(rebuild, first) is first
    assert _common.run_weight_function(rebuild, second) is second
    assert calls == [(first.device, None), (first.device, "torch"), (first.device, "torch"), (second.device, None)]


def test_lora_extraction_caps_rank_and_keeps_convolution_shape():
    method = merge_methods.resolve("extract_lora_sd1-kohya_lycoris_lora").instantiate()
    weight = torch.arange(12.).reshape(6, 2, 1, 1)
    relation = RealizedKeyRelation(("up", "down", "alpha"), {})
    adapter = method(weight, 8, False, 2, None, key_relation=relation)

    assert adapter["up"].shape == (6, 2, 1, 1)
    assert adapter["alpha"].item() == 2
    torch.testing.assert_close(adapter["up"].flatten(1) @ adapter["down"].flatten(1), weight.flatten(1))


def test_kolors_applies_diffusers_lora_with_inferred_adapter_config(tmp_path):
    module = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k"
    key = f"{module}.weight"
    weight = torch.ones(640, 640)
    down = torch.full((2, 640), 0.5)
    up = torch.ones(640, 2)
    base = sd_mecha.model({key: weight}, "kolors-diffusers")
    adapter_path = tmp_path / "kolors-lora.safetensors"
    save_file({
        f"unet.{module}.lora.down.weight": down,
        f"unet.{module}.lora.up.weight": up,
    }, adapter_path)

    recipe = sd_mecha.apply_lycoris(base, sd_mecha.model(adapter_path))
    assert recipe.merge_method.identifier == "apply_lycoris_kolors-diffusers_lycoris_lora"
    torch.testing.assert_close(merge(recipe)[key], weight + up @ down)


def test_sparse_lora_merge_plans_only_the_available_weight():
    weight = torch.eye(4).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight}, CONFIG, "delta")
    method = merge_methods.resolve("extract_lora_sd1-kohya_lycoris_lora")
    recipe = method(delta, 4, False, 2, None)
    extraction = method.__wrapped__
    probe_sizes = []
    output_groups = PropagatableKeyVisitor._root_output_groups

    def count_groups(visitor, node):
        groups = output_groups(visitor, node)
        probe_sizes.append(len(groups))
        return groups

    with (
        patch.object(extraction, "get_output_keys", wraps=extraction.get_output_keys) as get_output_keys,
        patch.object(PropagatableKeyVisitor, "_root_output_groups", count_groups),
    ):
        result = merge(recipe)

    get_output_keys.assert_called_once_with(WEIGHT)
    assert probe_sizes == [3]
    prefix = "lycoris_vae_post_quant_conv"
    up = result[f"{prefix}.lora_up.weight"]
    down = result[f"{prefix}.lora_down.weight"]
    assert result[f"{prefix}.alpha"].item() == 4
    torch.testing.assert_close(up.flatten(1) @ down.flatten(1), weight.flatten(1))


@pytest.mark.filterwarnings("error::UserWarning")
def test_norm_extraction_and_delta_conversion():
    bias = torch.arange(4.)
    delta = sd_mecha.model({BIAS: bias}, CONFIG, "delta")
    adapter = sd_mecha.lycoris.resolve("norm")(delta)
    result = merge(adapter)
    torch.testing.assert_close(result["lycoris_vae_post_quant_conv.b_norm"], bias)

    converted = merge(sd_mecha.convert(adapter, CONFIG), "delta")
    torch.testing.assert_close(converted[BIAS], bias)


@pytest.mark.parametrize("tucker", [False, True])
def test_loha_applies_a_scaled_hadamard_update(tucker):
    prefix = "lycoris_vae_post_quant_conv"
    left = torch.arange(1.0, 9.0).reshape(4, 2)
    right = torch.arange(1.0, 9.0).reshape(2, 4)
    factors = {
        "hada_w1_a": left.T if tucker else left,
        "hada_w1_b": right,
        "hada_w2_a": torch.ones_like(left.T if tucker else left),
        "hada_w2_b": torch.ones_like(right),
        "alpha": torch.tensor(0.5),
    }
    if tucker:
        factors["hada_t1"] = torch.eye(2).reshape(2, 2, 1, 1)
        factors["hada_t2"] = factors["hada_t1"].clone()
    weights = {f"{prefix}.{suffix}": tensor.contiguous() for suffix, tensor in factors.items()}
    delta = (0.5 * (left @ right)).reshape(4, 4, 1, 1)
    base = sd_mecha.model({WEIGHT: torch.ones_like(delta)}, CONFIG)
    adapter = sd_mecha.model(weights, f"{CONFIG}_lycoris_lora")

    torch.testing.assert_close(merge(sd_mecha.apply_lycoris(base, adapter))[WEIGHT], 1 + delta)
    torch.testing.assert_close(merge(sd_mecha.convert(adapter, CONFIG), "delta")[WEIGHT], delta)


@pytest.mark.parametrize("rescaled", [False, True])
@pytest.mark.parametrize("constraint", [None, 0.05])
def test_boft_applies_the_upstream_weight_transform(rescaled, constraint):
    layer = torch.nn.Conv2d(4, 4, 1)
    upstream = ButterflyOFTModule(
        "lora_vae_post_quant_conv", layer, lora_dim=2, constraint=constraint or 0, rescaled=rescaled,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.arange(16.0).reshape_as(layer.weight) / 16)
        upstream.oft_blocks[:, :, 0, 1] = torch.tensor([[0.2, -0.3], [0.4, 0.1]])
        if rescaled:
            upstream.rescale.copy_(torch.tensor([0.8, 1.1, 0.7, 1.2]).reshape_as(upstream.rescale))
        expected, _ = upstream.get_merged_weight()

    weights = upstream.state_dict()
    if constraint is None:
        del weights["alpha"]
    adapter = sd_mecha.model({f"lora_vae_post_quant_conv.{key}": value for key, value in weights.items()})
    base = sd_mecha.model({WEIGHT: layer.weight.detach(), BIAS: layer.bias.detach()}, CONFIG)
    result = merge(sd_mecha.apply_lycoris(base, adapter))

    torch.testing.assert_close(result[WEIGHT], expected)
    torch.testing.assert_close(result[BIAS], layer.bias.detach())


def test_boft_distinguishes_diag_oft_blocks():
    base = sd_mecha.model({WEIGHT: torch.ones(4, 4, 1, 1)}, CONFIG)
    adapter = sd_mecha.model({"lycoris_vae_post_quant_conv.oft_blocks": torch.zeros(2, 2, 2)})
    with pytest.raises(ValueError, match="BOFT expects 4-D oft_blocks.*Diag-OFT adapters use 3-D"):
        merge(sd_mecha.apply_lycoris(base, adapter))


@pytest.mark.parametrize("solver", ["svd_sweeps", "lbfgs"])
def test_boft_extraction_round_trips_and_uses_only_shared_weights(tmp_path, solver):
    layer = torch.nn.Conv2d(4, 4, 1)
    reference = ButterflyOFTModule("lora_vae_post_quant_conv", layer, lora_dim=2, rescaled=True)
    weights = torch.eye(4).reshape(4, 4, 1, 1)
    with torch.no_grad():
        layer.weight.copy_(weights)
        reference.oft_blocks[:, :, 0, 1] = torch.tensor([[0.09, -0.05], [0.06, 0.04]])
        reference.rescale.copy_(torch.tensor([1.02, 0.99, 1.04, 0.98]).reshape_as(reference.rescale))
        target_weights, _ = reference.get_merged_weight()

    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    # 320 output channels cannot use a two-channel block with a power-of-two block count.
    skipped = {"unet.conv_in.weight": torch.zeros(320, 4, 3, 3)}
    save_file({WEIGHT: weights, BIAS: torch.zeros(4), "vae.quant_conv.weight": torch.zeros(8, 8, 1, 1), **skipped}, base_path)
    save_file({
        WEIGHT: target_weights, BIAS: torch.ones(4), "vae.encoder.conv_in.weight": torch.zeros(128, 3, 3, 3), **skipped,
    }, target_path)
    base = sd_mecha.model(base_path, CONFIG)
    target = sd_mecha.model(target_path, CONFIG)
    extract = sd_mecha.lycoris.resolve("boft")
    assert extract is merge_methods.resolve("extract_boft")
    assert get_args(extract.get_input_types().args[-1]) == ("svd_sweeps", "lbfgs")
    solver_kwargs = {} if solver == "svd_sweeps" else {"solver": solver}
    serialized = sd_mecha.serialize(extract(base, target, block_size=2, **solver_kwargs))
    method_id = f"extract_boft_{CONFIG}_lycoris_lora"
    assert f'merge "{method_id}"' in serialized
    with pytest.raises(ValueError, match="Invalid value 'typo' for parameter 'solver'"):
        sd_mecha.deserialize(serialized.replace(f'"{solver}"', '"typo"'))
    legacy = serialized.replace(f'"{method_id}"', f'"extract_boft_{solver}_{CONFIG}_lycoris_lora"')
    with pytest.warns(FutureWarning, match="Recipe merge method .* is deprecated"):
        assert sd_mecha.serialize(sd_mecha.deserialize(legacy)) == serialized
    recipe = sd_mecha.deserialize(serialized)
    adapter_path = tmp_path / "adapter.safetensors"
    sd_mecha.merge(recipe, output=adapter_path, threads=0, output_dtype=torch.float32)
    adapter = load_file(adapter_path)

    prefix = "lycoris_vae_post_quant_conv"
    assert set(adapter) == {
        f"{prefix}.oft_blocks", f"{prefix}.rescale", f"{prefix}.alpha",
    }
    assert adapter[f"{prefix}.oft_blocks"].shape == (2, 2, 2, 2)
    assert adapter[f"{prefix}.rescale"].shape == (4, 1, 1, 1)
    assert adapter[f"{prefix}.alpha"].item() == 0
    merged = merge(sd_mecha.apply_lycoris(base, sd_mecha.model(adapter_path)))
    torch.testing.assert_close(merged[WEIGHT], target_weights, atol=2e-3, rtol=0)
    torch.testing.assert_close(merged[BIAS], torch.zeros(4))


def test_boft_selects_the_solver_per_key():
    other_key = "vae.quant_conv.weight"
    base_weights = {key: torch.eye(4).reshape(4, 4, 1, 1) for key in (WEIGHT, other_key)}
    base = sd_mecha.model(base_weights, CONFIG)
    target = sd_mecha.model({key: value * 2 for key, value in base_weights.items()}, CONFIG)
    recipe = sd_mecha.lycoris.resolve("boft")(
        base, target, block_size=2, solver={WEIGHT: "svd_sweeps", other_key: "lbfgs"},
    )

    blocks = torch.zeros(1, 2, 2, 2)
    gains = torch.ones(4, 1, 1, 1)
    with (
        patch("sd_mecha.extensions.builtin.lycoris.boft._fit_boft", return_value=(blocks, gains)) as svd,
        patch("sd_mecha.extensions.builtin.lycoris.boft._fit_boft_lbfgs", return_value=(blocks, 2 * gains)) as lbfgs,
    ):
        result = merge(recipe)

    svd.assert_called_once()
    lbfgs.assert_called_once()
    torch.testing.assert_close(result["lycoris_vae_post_quant_conv.rescale"], gains)
    torch.testing.assert_close(result["lycoris_vae_quant_conv.rescale"], 2 * gains)


def test_boft_extraction_can_start_from_zero_channel_gains():
    base_weights = torch.eye(4).reshape(4, 4, 1, 1)
    quarter_turn = torch.tensor([[0., -1.], [1., 0.]])
    target_weights = torch.block_diag(quarter_turn, quarter_turn).reshape_as(base_weights)
    base = sd_mecha.model({WEIGHT: base_weights}, CONFIG)
    target = sd_mecha.model({WEIGHT: target_weights}, CONFIG)
    extract = sd_mecha.lycoris.resolve("boft")
    adapter = merge(extract(base, target, block_size=2, stages=1))
    assert adapter["lycoris_vae_post_quant_conv.oft_blocks"].shape[0] == 1
    merged = merge(sd_mecha.apply_lycoris(base, sd_mecha.model(adapter, f"{CONFIG}_lycoris_lora")))
    torch.testing.assert_close(merged[WEIGHT], target_weights)


def test_boft_extraction_keeps_half_turn_generator_finite():
    identity = torch.eye(4)
    half_turn = torch.diag(torch.tensor([-1., -1., 1., 1.]))
    blocks = _cayley_blocks(half_turn[None, None])[0, 0]
    assert blocks.isfinite().all()
    generator = blocks - blocks.mT
    applied = torch.linalg.solve(identity - generator, identity + generator)
    torch.testing.assert_close(applied, half_turn, atol=4e-3, rtol=0)


def test_full_delta_applies_to_weights_and_biases():
    prefix = "lycoris_vae_post_quant_conv"
    weight = torch.arange(16.0).reshape(4, 4, 1, 1)
    bias = torch.arange(4.0)
    base = sd_mecha.model({WEIGHT: weight, BIAS: bias}, CONFIG)
    adapter = sd_mecha.model({
        f"{prefix}.diff": weight + 1,
        f"{prefix}.diff_b": bias + 2,
    }, f"{CONFIG}_lycoris_lora")

    result = merge(sd_mecha.apply_lycoris(base, adapter))
    torch.testing.assert_close(result[WEIGHT], 2 * weight + 1)
    torch.testing.assert_close(result[BIAS], 2 * bias + 2)


@pytest.mark.parametrize("tucker", [False, True])
def test_factorized_lokr_keeps_convolution_axes_and_alpha(tucker):
    key = "unet.conv_in.weight"
    prefix = "lycoris_unet_conv_in"
    w1 = torch.arange(1., 41.).reshape(20, 2)
    left = torch.arange(1., 33.).reshape(16, 2)
    if tucker:
        right = torch.eye(2)
        core = torch.arange(1., 37.).reshape(2, 2, 3, 3)
        w2 = torch.einsum("ijxy,oi,jp->opxy", core, left, right)
        factors = {"lokr_w2_a": left.T.contiguous(), "lokr_w2_b": right, "lokr_t2": core}
    else:
        right = torch.arange(1., 37.).reshape(2, 2, 3, 3)
        w2 = (left @ right.flatten(1)).reshape(16, 2, 3, 3)
        factors = {"lokr_w2_a": left, "lokr_w2_b": right}
    delta = torch.kron(w1[..., None, None], w2.contiguous()) * 0.5
    base = sd_mecha.model({key: torch.ones_like(delta)}, CONFIG)
    adapter = sd_mecha.model({
        f"{prefix}.lokr_w1": w1,
        f"{prefix}.alpha": torch.tensor(1.),
        **{f"{prefix}.{suffix}": tensor for suffix, tensor in factors.items()},
    }, f"{CONFIG}_lycoris_lora")

    torch.testing.assert_close(merge(sd_mecha.apply_lycoris(base, adapter))[key], 1 + delta)


def test_lora_tucker_convolution_preserves_alpha_and_factor_order():
    key = "vae.decoder.conv_out.weight"
    prefix = "lycoris_vae_decoder_conv_out"
    up = torch.arange(1., 7.).reshape(3, 2, 1, 1)
    mid = torch.arange(1., 37.).reshape(2, 2, 3, 3)
    down = torch.arange(1., 257.).reshape(2, 128, 1, 1)
    delta = torch.einsum("oi,ijxy,jp->opxy", up.flatten(1), mid, down.flatten(1)) * 0.5
    base = sd_mecha.model({key: torch.ones_like(delta)}, CONFIG)
    adapter = sd_mecha.model({
        f"{prefix}.lora_up.weight": up,
        f"{prefix}.lora_mid.weight": mid,
        f"{prefix}.lora_down.weight": down,
        f"{prefix}.alpha": torch.tensor(1.),
    }, f"{CONFIG}_lycoris_lora")

    torch.testing.assert_close(merge(sd_mecha.apply_lycoris(base, adapter))[key], 1 + delta)


def test_rank_one_lora_accepts_a_squeezed_up_factor():
    prefix = "lycoris_vae_post_quant_conv"
    up = torch.arange(1., 5.)
    down = torch.arange(2., 6.).reshape(1, 4, 1, 1)
    delta = (up[:, None] @ down.flatten(1)).reshape(4, 4, 1, 1)
    base = sd_mecha.model({WEIGHT: torch.ones_like(delta)}, CONFIG)
    adapter = sd_mecha.model({
        f"{prefix}.lora_up.weight": up,
        f"{prefix}.lora_down.weight": down,
    }, f"{CONFIG}_lycoris_lora")

    torch.testing.assert_close(merge(sd_mecha.apply_lycoris(base, adapter))[WEIGHT], 1 + delta)


def test_loha_tucker_accepts_broadcastable_spatial_cores():
    key = "vae.decoder.conv_out.weight"
    prefix = "lycoris_vae_decoder_conv_out"
    up = torch.ones(1, 3)
    down = torch.ones(1, 128)
    spatial = torch.arange(1., 10.).reshape(1, 1, 3, 3)
    delta = spatial.expand(3, 128, 3, 3)
    base = sd_mecha.model({key: torch.ones_like(delta)}, CONFIG)
    adapter = sd_mecha.model({
        f"{prefix}.hada_w1_a": up,
        f"{prefix}.hada_w1_b": down,
        f"{prefix}.hada_w2_a": up,
        f"{prefix}.hada_w2_b": down,
        f"{prefix}.hada_t1": torch.ones(1, 1, 1, 1),
        f"{prefix}.hada_t2": spatial,
    }, f"{CONFIG}_lycoris_lora")

    with pytest.warns(UserWarning, match="LoHa factors with different output shapes"):
        result = merge(sd_mecha.apply_lycoris(base, adapter))
    torch.testing.assert_close(result[key], 1 + delta)


@pytest.mark.parametrize(
    "up_key, down_key",
    [
        ("lora_unet_final_layer_linear.lora_up.weight", "lora_unet_final_layer_linear.lora_down.weight"),
        ("diffusion_model.final_layer.linear.lora_B.weight", "diffusion_model.final_layer.linear.lora_A.weight"),
    ],
)
def test_anima_adapter_reads_lora_factor_aliases(tmp_path, up_key, down_key):
    checkpoint = tmp_path / "anima-lora.safetensors"
    save_file({up_key: torch.full((64, 1), 2.0), down_key: torch.ones((1, 2048))}, checkpoint)
    key = "net.final_layer.linear.weight"
    base = sd_mecha.model({key: torch.ones(64, 2048)}, "anima-diffusion_pipe")
    adapter = sd_mecha.model(checkpoint)

    result = merge(sd_mecha.apply_lycoris(base, adapter))

    torch.testing.assert_close(result[key], torch.full((64, 2048), 3.0))


@pytest.mark.parametrize(
    "stem, up, down",
    [
        ("text_encoders_qwen3_06b_transformer_model_layers_0_self_attn_q_proj", "lora_up", "lora_down"),
        ("text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj", "lora_up", "lora_down"),
        ("text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj", "lora_B", "lora_A"),
        ("lora_te_layers_0_self_attn_q_proj", "lora_up", "lora_down"),
    ],
)
def test_anima_qwen_lora_resolves_dotted_and_flattened_stems(tmp_path, stem, up, down):
    checkpoint = tmp_path / "anima-qwen-lora.safetensors"
    save_file({
        f"{stem}.{up}.weight": torch.full((2048, 1), 2.0),
        f"{stem}.{down}.weight": torch.ones((1, 1024)),
    }, checkpoint)
    key = "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.weight"
    base = sd_mecha.model({key: torch.ones(2048, 1024)}, "anima-diffusion_pipe")

    result = merge(sd_mecha.apply_lycoris(base, sd_mecha.model(checkpoint)))

    torch.testing.assert_close(result[key], torch.full((2048, 1024), 3.0))


@pytest.mark.parametrize("entry_point", ["apply_lycoris", "add_difference", "convert"])
def test_moss_image_lora_applies_to_comfyui_qkv_via_openmoss(tmp_path, entry_point):
    checkpoint = tmp_path / "moss-image-edit-lora.safetensors"
    save_file({
        "layers.0.proj.q.lora_A": torch.ones((1, 3072)),
        "layers.0.proj.q.lora_B": torch.full((3072, 1), 10.0),
        "layers.0.proj.v.lora_A.default.weight": torch.ones((1, 3072)),
        "layers.0.proj.v.lora_B.default.weight": torch.full((3072, 1), 20.0),
    }, checkpoint)
    key = "layers.0.proj.qkv.weight"
    qkv = torch.arange(1.0, 4.0).repeat_interleave(3072).unsqueeze(1).expand(-1, 3072)
    rope = torch.arange(16.0)
    base = sd_mecha.model({key: qkv, "rope.bands": rope}, "moss_image-comfyui")

    adapter = sd_mecha.model(checkpoint)
    if entry_point == "apply_lycoris":
        applied = sd_mecha.apply_lycoris(base, adapter)
        assert applied.bound_args.arguments["base"].merge_method.identifier == "convert_moss_image_comfyui_to_openmoss"
        recipe = sd_mecha.convert(applied, base)
    elif entry_point == "add_difference":
        recipe = sd_mecha.add_difference(base, adapter)
    else:
        recipe = (base + sd_mecha.convert(adapter, base)) | base

    result = sd_mecha.merge(
        recipe, output_dtype=torch.float32, threads=0, check_finite_output=False, omit_non_finite_inputs=False,
    )

    for projection, expected in zip(result[key].chunk(3), (11.0, 2.0, 23.0), strict=True):
        sample = projection[:, :4]
        torch.testing.assert_close(sample, torch.full_like(sample, expected))
    torch.testing.assert_close(result["rope.bands"], rope)


@pytest.mark.parametrize("config", ["sdxl-kohya", "sdxl-kohya_but_diffusers"])
def test_sdxl_clip_g_packing_preserves_sparse_projection_deltas(config):
    key = "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"
    prefix = "lycoris_te2_text_model_encoder_layers_0_self_attn"
    adapter = sd_mecha.model({
        f"{prefix}_{part}_proj.lora_up.weight": torch.full((1280, 1), scale)
        for part, scale in (("q", 2.), ("v", 3.))
    } | {
        f"{prefix}_{part}_proj.lora_down.weight": torch.ones(1, 1280)
        for part in ("q", "v")
    }, f"{config}_lycoris_lora")

    result = merge(sd_mecha.convert(adapter, "sdxl-sgm"), "delta")
    for projection, expected in zip(result[key].chunk(3), (2., 0., 3.), strict=True):
        torch.testing.assert_close(projection[:, :4], torch.full_like(projection[:, :4], expected))

    partial_weights = sd_mecha.model({
        f"te2.text_model.encoder.layers.0.self_attn.{part}_proj.weight": torch.ones(1280, 1280)
        for part in ("q", "v")
    }, config)
    with sd_mecha.open_graph(sd_mecha.convert(partial_weights, "sdxl-sgm")) as graph:
        plan = graph.finalize_with_keys(check_mandatory_keys=False)
    assert key not in plan.node_to_keys.get(plan.root, ())


def test_sdxl_diffusers_key_layout_retains_vae_and_text_encoders():
    tensors = {
        "first_stage_model.decoder.conv_out.bias": torch.arange(3.),
        "conditioner.embedders.0.transformer.text_model.final_layer_norm.bias": torch.ones(768),
        "conditioner.embedders.1.model.ln_final.bias": torch.ones(1280),
        "model.diffusion_model.out.2.bias": torch.arange(4.),
    }
    recipe = sd_mecha.convert(sd_mecha.model(tensors, "sdxl-sgm"), "sdxl-kohya_but_diffusers")

    result = merge(recipe)

    expected_keys = (
        "first_stage_model.decoder.conv_out.bias",
        "te1.text_model.final_layer_norm.bias",
        "te2.text_model.final_layer_norm.bias",
        "unet.conv_out.bias",
    )
    for key, tensor in zip(expected_keys, tensors.values(), strict=True):
        torch.testing.assert_close(result[key], tensor)


def test_adapter_key_aliases_merge_to_canonical_keys():
    key = "lora_vae_post_quant_conv.lora_up.weight"
    tensor = torch.arange(4.0).reshape(4, 1, 1, 1)
    adapter = sd_mecha.model({key: tensor}, f"{CONFIG}_lycoris_lora")
    lycoris_key = "lycoris_vae_post_quant_conv.lora_up.weight"
    torch.testing.assert_close(merge(adapter)[lycoris_key], tensor)


@pytest.mark.filterwarnings("error::UserWarning")
def test_to_base_expands_the_conversion_from_lycoris(tmp_path):
    tensor = torch.arange(16.0).reshape(4, 4, 1, 1)
    checkpoint = tmp_path / "adapter.safetensors"
    save_file({"lora_vae_post_quant_conv.diff": tensor}, checkpoint)
    adapter = sd_mecha.model(checkpoint, f"{CONFIG}_lycoris_lora")

    recipe = sd_mecha.to_base(adapter)
    assert recipe.merge_method.identifier == f"convert_'{CONFIG}_lycoris_lora'_to_base"
    assert recipe.bound_args.args[0] is adapter
    assert sd_mecha.to_base(recipe) is recipe

    result = merge(sd_mecha.deserialize(sd_mecha.serialize(recipe)), "delta")
    torch.testing.assert_close(result[WEIGHT], tensor)


@pytest.mark.parametrize("adapter_config", [None, "anima-diffusion_pipe_lycoris_lora"])
@pytest.mark.parametrize("fallback_to_a", [True, False])
@pytest.mark.filterwarnings("error::UserWarning")
def test_add_difference_converts_an_adapter_to_a_delta(tmp_path, adapter_config, fallback_to_a):
    key = "net.final_layer.linear.weight"
    other_key = "net.pos_embedder.seq"
    other = torch.arange(256.0)
    checkpoint = tmp_path / "anima-adapter.safetensors"
    save_file({
        "lora_unet_final_layer_linear.lora_up.weight": torch.full((64, 1), 2.0),
        "lora_unet_final_layer_linear.lora_down.weight": torch.ones((1, 2048)),
    }, checkpoint)
    base = sd_mecha.model({key: torch.ones(64, 2048), other_key: other}, "anima-diffusion_pipe")

    fallback = {} if fallback_to_a else {"fallback_to_a": False}
    result = merge(sd_mecha.add_difference(base, sd_mecha.model(checkpoint, adapter_config), **fallback))

    torch.testing.assert_close(result[key], torch.full((64, 2048), 3.0))
    if fallback_to_a:
        torch.testing.assert_close(result[other_key], other)
    else:
        assert other_key not in result


@pytest.mark.parametrize("mixed", [False, True])
def test_delta_conversion_warns_when_it_skips_boft_updates(tmp_path, mixed):
    weight = torch.arange(16.).reshape(4, 4, 1, 1) / 16
    bias = torch.zeros(4)
    base = sd_mecha.model({WEIGHT: weight, BIAS: bias}, CONFIG)
    blocks = torch.zeros(2, 2, 2, 2)
    blocks[0, 0, 0, 1] = 0.2
    weights = {"lora_vae_post_quant_conv.oft_blocks": blocks}
    if mixed:
        weights["lora_vae_post_quant_conv.diff_b"] = torch.ones(4)
    checkpoint = tmp_path / "adapter.safetensors"
    save_file(weights, checkpoint)
    adapter = sd_mecha.model(checkpoint, f"{CONFIG}_lycoris_lora")

    expected_bias = torch.ones(4) if mixed else bias
    with pytest.warns(UserWarning, match="conversion skipped BOFT updates") as emitted:
        result = merge(sd_mecha.add_difference(base, adapter))
    assert len(emitted) == 1
    torch.testing.assert_close(result[WEIGHT], weight)
    torch.testing.assert_close(result[BIAS], expected_bias)

    if mixed:
        with pytest.warns(UserWarning, match="conversion skipped BOFT updates"):
            delta = merge(sd_mecha.to_base(adapter), "delta")
        assert WEIGHT not in delta
        torch.testing.assert_close(delta[BIAS], expected_bias)
    else:
        with pytest.warns(UserWarning, match="conversion skipped BOFT updates"):
            with pytest.raises(RuntimeError, match="produces no output keys"):
                merge(sd_mecha.to_base(adapter), "delta")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        applied = merge(sd_mecha.apply_lycoris(base, adapter))
    assert not torch.equal(applied[WEIGHT], weight)
    torch.testing.assert_close(applied[BIAS], expected_bias)


@pytest.mark.filterwarnings("error::UserWarning")
def test_delta_conversion_ignores_boft_keys_filtered_by_an_input_recipe():
    key = "lycoris_vae_post_quant_conv.diff_b"

    @sd_mecha.merge_method(register=False)
    class keep_bias:
        @staticmethod
        def map_keys(builder):
            builder[key] = builder.keys[key]

        def __call__(self, value: sd_mecha.Parameter(torch.Tensor)) -> sd_mecha.Return(torch.Tensor):
            return value

    bias = torch.ones(4)
    source = sd_mecha.model({
        key: bias,
        "lycoris_vae_post_quant_conv.oft_blocks": torch.zeros(2, 2, 2, 2),
    }, f"{CONFIG}_lycoris_lora")
    delta = merge(sd_mecha.to_base(keep_bias(source)), "delta")
    torch.testing.assert_close(delta[BIAS], bias)


def test_add_difference_keeps_structural_inputs():
    key = "user.weight"
    base = sd_mecha.model({key: torch.ones(1)})
    delta = sd_mecha.model({key: torch.full((1,), 2.0)}, merge_space="delta")

    torch.testing.assert_close(merge(base + delta)[key], torch.full((1,), 3.0))


def test_lokr_extraction_default_round_trips_through_application():
    w1 = torch.diag(torch.tensor([1., 2.]))
    w2 = torch.tensor([[3., 1.], [-2., 4.]])
    weight = torch.kron(w1, w2).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight}, CONFIG, "delta")
    base = sd_mecha.model({WEIGHT: torch.zeros_like(weight)}, CONFIG)
    adapter = sd_mecha.lycoris.resolve("lokr")(delta)

    extracted = merge(adapter)
    assert extracted["lycoris_vae_post_quant_conv.lokr_w1"].ndim == 2
    assert extracted["lycoris_vae_post_quant_conv.alpha"].item() == 1.0
    torch.testing.assert_close(merge(sd_mecha.apply_lycoris(base, adapter))[WEIGHT], weight)


@pytest.mark.parametrize("select_dims", [None, lokr_dims, lokr_dims_lycoris])
def test_truncate_kronecker_dimensions_and_bias_passthrough(select_dims):
    bias = torch.arange(4.)
    weight = torch.kron(torch.eye(2), torch.ones((2, 2))).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight, BIAS: bias}, CONFIG, "delta")
    dims = select_dims(delta) if select_dims is not None else None
    result = merge(truncate_kronecker(delta, dims, use_approximate_basis=False), "delta")

    torch.testing.assert_close(result[WEIGHT], weight)
    torch.testing.assert_close(result[BIAS], bias)


def test_truncate_kronecker_requires_a_dimension_matrix():
    with pytest.raises(ValueError, match="expected 2x2 preference"):
        truncate_kronecker.__wrapped__(torch.eye(4), torch.tensor(0.5))


@pytest.mark.parametrize("method", ["truncate_kronecker", "lora", "lokr"])
def test_kronecker_and_adapter_caches_honor_storage_settings(method):
    weight = torch.eye(4).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight}, CONFIG, "delta")
    if method == "truncate_kronecker":
        recipe = sd_mecha.truncate_kronecker(delta, use_approximate_basis=False)
        space, entry_key, factor = "delta", WEIGHT, "w1"
    elif method == "lora":
        recipe = sd_mecha.lycoris.resolve("lora")(delta, rank=1, use_approximate_basis=False)
        space, entry_key, factor = "weight", "lycoris_vae_post_quant_conv.lora_up.weight", "u"
    else:
        recipe = sd_mecha.lycoris.resolve("lokr")(delta)
        space, entry_key, factor = "weight", "lycoris_vae_post_quant_conv.lokr_w1", "w1"

    cache = recipe.merge_method.create_cache(dtype=torch.float64)
    options = dict(cache={recipe: cache}, strict_merge_space=space, output_dtype=torch.float32, threads=0)
    sd_mecha.merge(recipe, **options)
    cached = cache[entry_key][factor]
    assert cached.dtype == torch.float64
    sd_mecha.merge(recipe, **options)
    assert cache[entry_key][factor] is cached

    cache["dtype"] = torch.float32
    sd_mecha.merge(recipe, **options)
    assert cache[entry_key][factor].dtype == torch.float32
    assert cache["dtype"] == torch.float32


def test_lokr_recipe_format_version(tmp_path):
    checkpoint = tmp_path / "delta.safetensors"
    save_file({WEIGHT: torch.ones((4, 4, 1, 1))}, checkpoint)
    recipe = sd_mecha.lycoris.resolve("lokr")(sd_mecha.model(checkpoint, CONFIG, "delta"))
    serialized = sd_mecha.serialize(recipe)

    assert serialized.splitlines()[0] == "version 0.1.0"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        restored = sd_mecha.deserialize(serialized)
    assert restored.merge_method.get_identifier() == "extract_lokr_sd1-kohya_lycoris_lora"


@pytest.mark.parametrize(
    "old_name, arguments, algorithm, ratio",
    [
        ("extract_lycoris_lora", "", "lora", None),
        ("extract_kohya_lora", " rank=2", "lora", None),
        ("extract_lora_'sd1-kohya_lycoris_lora'", " 2 false 2 null", "lora", None),
        ("extract_lycoris_lokr", "", "lokr", 0.5),
        ("extract_kohya_lokr", " kronecker_ratio=0.25", "lokr", 0.25),
        ("extract_lokr_'sd1-kohya_kohya_lora'", " 0.75", "lokr", 0.75),
        ("extract_lora_sd1-kohya_kohya_lora", " 2 false 2 null", "lora", None),
    ],
)
def test_deserialize_deprecated_extraction_method(tmp_path, old_name, arguments, algorithm, ratio):
    checkpoint = tmp_path / "delta.safetensors"
    save_file({WEIGHT: torch.ones((4, 4, 1, 1))}, checkpoint)
    source = (
        "version 0.1.0\n"
        f'model "{checkpoint}" model_config="{CONFIG}" merge_space="delta"\n'
        f'merge "{old_name}" &0{arguments}'
    )
    replacement = f"extract_{algorithm}_{CONFIG}_lycoris_lora"
    with pytest.warns(FutureWarning, match=r"removed in sd-mecha 1\.3\.0") as caught:
        recipe = sd_mecha.deserialize(source)

    assert old_name in str(caught[0].message)
    assert replacement in str(caught[0].message)
    assert recipe.merge_method.get_identifier() == replacement
    if ratio is not None:
        dims = recipe.bound_args.arguments["dims"]
        expected_dims = lokr_dims(recipe.bound_args.arguments["base"], ratio)
        assert sd_mecha.serialize(dims, finalize=False) == sd_mecha.serialize(expected_dims, finalize=False)

    serialized = sd_mecha.serialize(recipe)
    assert old_name not in serialized
    assert replacement in serialized
    if old_name.startswith("extract_kohya_"):
        tensor_keys = [key for key, value in merge(recipe).items() if isinstance(value, torch.Tensor)]
        assert tensor_keys and all(key.startswith("lycoris_") for key in tensor_keys)
