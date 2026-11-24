from itertools import combinations

import torch


def pack_key_requirements(builder, source_keys):
    """Select the complete projection group, or the largest available subset in delta space."""
    requirements = builder.keys[source_keys]
    if getattr(builder, "return").merge_space == "delta":
        for size in range(len(source_keys) - 1, 0, -1):
            for available in combinations(source_keys, size):
                requirements |= builder.keys[available]
    return requirements


def convert_n_to_1(source, selected_keys, transpose=False, *, source_keys):
    if len(selected_keys) == len(source_keys):
        res = [source[key] for key in source_keys]
    else:
        selected = {key: source[key] for key in selected_keys}
        template = next(iter(selected.values()))
        metadata = source.model_config.keys()
        res = [selected[key] if key in selected else template.new_zeros(metadata[key].shape) for key in source_keys]
    if len(res) > 1:
        res = torch.cat(res, dim=0)
    else:
        res = res[0]

    if transpose:
        res = res.T

    return res


def convert_1_to_n(sgm_value, kohya_keys, transpose=False):
    if transpose:
        sgm_value = sgm_value.T

    sgm_chunks = sgm_value.chunk(len(kohya_keys), dim=0)
    res = dict(zip(kohya_keys, sgm_chunks))
    return res
