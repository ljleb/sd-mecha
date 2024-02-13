import inspect
import logging

from sd_meh import merge_methods
from sd_meh.merge import NUM_TOTAL_BLOCKS
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS

MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
BETA_METHODS = [
    name
    for name, fn in MERGE_METHODS.items()
    if "beta" in inspect.getfullargspec(fn)[0]
]


def compute_weights(weights, base):
    if not weights:
        return [base] * NUM_TOTAL_BLOCKS

    if "," not in weights:
        return weights

    w_alpha = list(map(float, weights.split(",")))
    if len(w_alpha) == NUM_TOTAL_BLOCKS:
        return w_alpha


def assemble_weights_and_bases(preset, weights, base, greek_letter):
    logging.info(f"Assembling {greek_letter} w&b")
    if preset:
        logging.info(f"Using {preset} preset")
        base, *weights = BLOCK_WEIGHTS_PRESETS[preset]
    bases = {greek_letter: base}
    weights = {greek_letter: compute_weights(weights, base)}

    logging.info(f"base_{greek_letter}: {bases[greek_letter]}")
    logging.info(f"{greek_letter} weights: {weights[greek_letter]}")

    return weights, bases


def interpolate_presets(
    weights, bases, weights_b, bases_b, greek_letter, presets_lambda
):
    logging.info(f"Interpolating {greek_letter} w&b")
    for i, e in enumerate(weights[greek_letter]):
        weights[greek_letter][i] = (
            1 - presets_lambda
        ) * e + presets_lambda * weights_b[greek_letter][i]

    bases[greek_letter] = (1 - presets_lambda) * bases[
        greek_letter
    ] + presets_lambda * bases_b[greek_letter]

    logging.info(f"Interpolated base_{greek_letter}: {bases[greek_letter]}")
    logging.info(f"Interpolated {greek_letter} weights: {weights[greek_letter]}")

    return weights, bases


def weights_and_bases(
    merge_mode,
    weights_alpha,
    base_alpha,
    block_weights_preset_alpha,
    weights_beta,
    base_beta,
    block_weights_preset_beta,
    block_weights_preset_alpha_b,
    block_weights_preset_beta_b,
    presets_alpha_lambda,
    presets_beta_lambda,
):
    weights, bases = assemble_weights_and_bases(
        block_weights_preset_alpha,
        weights_alpha,
        base_alpha,
        "alpha",
    )

    if block_weights_preset_alpha_b:
        logging.info("Computing w&b for alpha interpolation")
        weights_b, bases_b = assemble_weights_and_bases(
            block_weights_preset_alpha_b,
            None,
            None,
            "alpha",
        )
        weights, bases = interpolate_presets(
            weights,
            bases,
            weights_b,
            bases_b,
            "alpha",
            presets_alpha_lambda,
        )

    if merge_mode in BETA_METHODS:
        weights_beta, bases_beta = assemble_weights_and_bases(
            block_weights_preset_beta,
            weights_beta,
            base_beta,
            "beta",
        )

        if block_weights_preset_beta_b:
            logging.info("Computing w&b for beta interpolation")
            weights_b, bases_b = assemble_weights_and_bases(
                block_weights_preset_beta_b,
                None,
                None,
                "beta",
            )
            weights, bases = interpolate_presets(
                weights,
                bases,
                weights_b,
                bases_b,
                "beta",
                presets_beta_lambda,
            )

        weights |= weights_beta
        bases |= bases_beta

    return weights, bases
