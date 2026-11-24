from __future__ import annotations

from typing import Literal

import torch
from lycoris.functional.boft import diff_weight
from lycoris.functional.general import power2factorization

from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.keys_map import ComposeObject, KeysAccessor, ReqExpr
from sd_mecha.streaming import StateDictKeyError

from ._common import BaseToLycoris, LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper, extraction_implementation_name, run_weight_function


def _rotate_rows(blocks, matrix, order, inverse_order):
    return (blocks @ matrix[:, order, :]).flatten(1, 2)[:, inverse_order, :]


def _apply_rotations(rotations, matrix, orders, inverse_orders):
    for blocks, order, inverse_order in zip(rotations, orders, inverse_orders):
        matrix = _rotate_rows(blocks, matrix, order, inverse_order)
    return matrix


def _sweep(rotations, correlation, orders, inverse_orders):
    # Remove the later stages on the left, then account for the already updated
    # earlier stages on the right. The remaining objective separates by block.
    after_stage = [correlation] * len(orders)
    for stage in range(len(orders) - 1, 0, -1):
        after_stage[stage - 1] = _rotate_rows(
            rotations[stage].mT, after_stage[stage], orders[stage], inverse_orders[stage],
        )

    for stage, order in enumerate(orders):
        current = after_stage[stage]
        for earlier in range(stage):
            current = _rotate_rows(rotations[earlier], current.mT, orders[earlier], inverse_orders[earlier]).mT
        block_correlation = current[:, order[..., None], order[:, None, :]]
        # Prefer the current rotation when the data leaves a direction unconstrained.
        strength = block_correlation.abs().amax(dim=(-1, -2), keepdim=True)
        tie_break = torch.finfo(block_correlation.dtype).eps * order.shape[-1] * strength
        u, _, vh = torch.linalg.svd(block_correlation + tie_break * rotations[stage])
        u[..., -1] *= torch.linalg.det(u @ vh).sign().unsqueeze(-1)
        rotations[stage] = torch.where(strength > 0, u @ vh, rotations[stage])


def _cayley_near_pi(rotation, identity):
    # A rotation at pi has no finite generator. Preserve the other rotation
    # planes and use a finite angle close to pi in the singular subspace.
    u, s, vh = torch.linalg.svd(rotation + identity)
    n_small = int((s < 0.004).sum())
    n_small = max(2, 2 * ((n_small + 1) // 2))
    regular = rotation.shape[-1] - n_small
    vectors = vh.mT
    generator = (vectors[:, :regular] / s[:regular]) @ (u[:, :regular].mT @ (rotation - identity))
    plane = vectors[:, regular::2] @ vectors[:, regular + 1::2].mT
    return generator + 512 * (plane - plane.mT)


def _cayley_blocks(rotations):
    identity = torch.eye(rotations.shape[-1], dtype=rotations.dtype, device=rotations.device)
    generator, info = torch.linalg.solve_ex(rotations + identity, rotations - identity)
    needs_limit = (info != 0) | (~generator.isfinite().all(dim=(-1, -2))) | (generator.abs().amax(dim=(-1, -2)) > 512)
    for index in needs_limit.nonzero().tolist():
        index = tuple(index)
        generator[index] = _cayley_near_pi(rotations[index], identity)
    # LyCORIS reconstructs the generator by subtracting the transpose.
    return (generator - generator.mT) * 0.25


def _fit_gains(rotation, rotation_gram, cross):
    dot = (rotation * cross).sum(-1, keepdim=True)
    norm = (rotation * rotation_gram).sum(-1, keepdim=True)
    return torch.where(norm > 0, dot / norm, 1.0)


def _stage_orders(width, block_size, stages, device):
    channels = torch.arange(width, device=device)
    orders = tuple(
        channels.unflatten(0, (-1, 2, 2 ** stage * (block_size // 2))).transpose(1, 2).reshape(-1, block_size)
        for stage in range(stages)
    )
    return orders, tuple(order.flatten().argsort() for order in orders)


def _fit_boft(base, target, block_size, stages):
    """Fit row gains and butterfly rotations with four block SVD sweeps."""
    if base.shape != target.shape:
        raise ValueError(f"BOFT extraction requires matching weight shapes, got {base.shape} and {target.shape}.")
    width = block_size * 2 ** (stages - 1)
    dtype = torch.float64 if torch.float64 in (base.dtype, target.dtype) else torch.float32
    matrix = base.flatten(1).unflatten(0, (-1, width)).to(dtype=dtype)
    target = target.reshape_as(matrix).to(dtype=dtype)
    gram = matrix @ matrix.mT
    cross = target @ matrix.mT

    orders, inverse_orders = _stage_orders(width, block_size, stages, base.device)
    block_identity = torch.eye(block_size, device=base.device, dtype=dtype).expand(matrix.shape[0], width // block_size, -1, -1)
    rotations = [block_identity] * stages
    identity = torch.eye(width, device=base.device, dtype=dtype).expand_as(gram)
    rotation = identity
    rotation_gram = gram

    for _ in range(4):
        gains = _fit_gains(rotation, rotation_gram, cross)
        # Subtracting max(gains²) * I removes a constant on orthogonal matrices.
        # The tangent of the remaining concave quadratic is a linear upper bound.
        squared = gains.square()
        max_squared = squared.amax(dim=-2, keepdim=True)
        surrogate = gains * cross + (max_squared - squared) * rotation_gram
        # With all gains zero the current loss is independent of the rotation.
        surrogate = torch.where(max_squared > 0, surrogate, cross)
        _sweep(rotations, surrogate, orders, inverse_orders)
        rotation = _apply_rotations(rotations, identity, orders, inverse_orders)
        rotation_gram = _apply_rotations(rotations, gram, orders, inverse_orders)

    gains = _fit_gains(rotation, rotation_gram, cross)
    blocks = _cayley_blocks(torch.stack(rotations).flatten(1, 2))
    return blocks, gains.reshape((base.shape[0],) + (1,) * (base.ndim - 1))


@torch.inference_mode(False)
@torch.enable_grad()
def _fit_boft_lbfgs(base, target, block_size, stages):
    """Refine the SVD fit by jointly optimizing the Cayley parameters and row gains."""
    width = block_size * 2 ** (stages - 1)
    with torch.no_grad():
        initial_blocks, initial_gains = _fit_boft(base, target, block_size, stages)
        matrix = base.double().flatten(1).unflatten(0, (-1, width))
        delta = (target.double() - base.double()).reshape_as(matrix)
        gram = matrix @ matrix.mT
        cross = delta @ matrix.mT
        orders, inverse_orders = _stage_orders(width, block_size, stages, base.device)
        block_identity = torch.eye(block_size, device=base.device, dtype=matrix.dtype)
        identity = torch.eye(width, device=base.device, dtype=matrix.dtype).expand_as(gram)

    # PyTorch's L-BFGS flattens gradients with view(), which needs contiguous parameters.
    blocks = initial_blocks.double().reshape(stages, matrix.shape[0], -1, block_size, block_size).contiguous().requires_grad_()
    gains = initial_gains.double().reshape(*matrix.shape[:2], 1).contiguous().requires_grad_()
    optimizer = torch.optim.LBFGS(
        (blocks, gains), max_iter=200, line_search_fn="strong_wolfe",
        tolerance_grad=1e-11, tolerance_change=1e-16,
    )

    def closure():
        optimizer.zero_grad()
        skew = blocks - blocks.mT
        rotations = torch.linalg.solve(block_identity - skew, block_identity + skew)
        rotation = _apply_rotations(rotations, identity, orders, inverse_orders)
        transform = gains * rotation - identity
        # Expand ||(D R - I) W - (T - W)||², omitting the term constant in D and R.
        loss = ((transform @ gram) * transform - 2 * transform * cross).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        blocks.detach().reshape_as(initial_blocks).to(initial_blocks.dtype),
        gains.detach().reshape_as(initial_gains).to(initial_gains.dtype),
    )


class boft(LycorisAlgorithm):
    """Apply a butterfly rotation and optional rescaling to the base weight's output channels."""

    suffixes = ("oft_blocks", "rescale", "alpha")

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> ReqExpr | None:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        blocks, rescale, alpha = lyco_keys
        return (
            keys[blocks] @ cls.apply &
            (keys[rescale] @ cls.with_rescale | keys[()]) &
            (keys[alpha] @ cls.with_constraint | keys[()])
        )

    class apply(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            blocks = state_dict.get_lyco_tensor("oft_blocks")
            if blocks.ndim != 4:
                raise ValueError(
                    f"BOFT expects 4-D oft_blocks for {state_dict.base_key!r}, got shape {tuple(blocks.shape)}. "
                    "Diag-OFT adapters use 3-D oft_blocks and require a different implementation."
                )

            base = state_dict.base_sd[state_dict.base_key]
            rescale = cls.get_rescale(state_dict, base.shape)
            constraint = cls.get_constraint(state_dict, base.shape[0])
            # TileLang's fused Cayley solve accepts blocks up to size 32.
            fallback_reason = "BOFT blocks larger than 32" if blocks.shape[-1] > 32 else None
            delta = run_weight_function(
                diff_weight, base, blocks.float(), rescale, constraint=constraint, fallback_reason=fallback_reason,
            )
            return base + delta.reshape(target_shape)

        @staticmethod
        def get_rescale(state_dict, shape):
            return None

        @staticmethod
        def get_constraint(state_dict, output_channels):
            return None

    class with_rescale(apply):
        @staticmethod
        def get_rescale(state_dict, shape):
            return state_dict.get_lyco_tensor("rescale").reshape((shape[0],) + (1,) * (len(shape) - 1))

    class with_constraint(apply):
        @staticmethod
        def get_constraint(state_dict, output_channels):
            return float(state_dict.get_lyco_tensor("alpha")) * output_channels

    @staticmethod
    def extract(
        base: Parameter(torch.Tensor, "weight"),
        target: Parameter(torch.Tensor, "weight"),
        block_size: Parameter(int, "param") = 10,
        stages: Parameter(int, "param") = 5,
        solver: Parameter(Literal["svd_sweeps", "lbfgs"], "param") = "svd_sweeps",
    ) -> Return(torch.Tensor, "weight"):
        """Fit a rescaled BOFT adapter that transforms base weights toward target weights.

        Args:
            base: The original weights to which the adapter will be applied.
            target: The weights to approximate, in the same model config as base.
            block_size: The largest allowed rotation block size. Mecha picks an even block size
                per weight so the number of blocks is a power of two. Keys without a valid
                layout at this size are skipped.
            stages: The number of butterfly stages. Mecha caps this at the maximum supported by
                the block layout for each weight.
            solver: Use `"svd_sweeps"` for four sweeps of block SVDs and gain updates, or
                `"lbfgs"` to refine that fit by jointly optimizing rotations and gains.

        Returns:
            A LyCORIS recipe with butterfly blocks and per-output-channel gains.
            Alpha is zero, which disables the rotation norm constraint.
        """
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        cls._register_extractor(lyco_config, base_config, "svd_sweeps", _fit_boft)
        cls._register_extractor(lyco_config, base_config, "lbfgs", _fit_boft_lbfgs)

    @classmethod
    def _register_extractor(cls, lyco_config, base_config, solver_name, fit):
        solver_type = Literal[solver_name]

        @merge_method(
            identifier=extraction_implementation_name(f"boft_{solver_name}", lyco_config),
            implements=cls.extract_method,
            globals=globals(),
            locals=locals(),
        )
        class BaseToBoft(BaseToLycoris):
            @staticmethod
            def get_output_keys(base_key: str):
                shape = base_config.keys()[base_key].shape
                if shape and shape[0] % 2 == 0:
                    return tuple(lyco_config.to_lycoris_keys(base_key, (cls,)))
                return ()

            def __call__(
                self,
                base: Parameter(torch.Tensor, "weight", base_config),
                target: Parameter(torch.Tensor, "weight", base_config),
                block_size: Parameter(int, "param", base_config),
                stages: Parameter(int, "param", base_config),
                solver: Parameter(solver_type, "param", base_config),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config):
                block_key, rescale_key, alpha_key = kwargs["key_relation"].outputs
                block_size, block_count = power2factorization(base.shape[0], min(max(2, block_size), base.shape[0]))
                if block_count == 0:
                    raise StateDictKeyError(block_key)
                stages = min(max(1, stages), block_count.bit_length())
                blocks, gains = fit(base, target, block_size, stages)
                return {block_key: blocks, rescale_key: gains, alpha_key: blocks.new_zeros(())}
