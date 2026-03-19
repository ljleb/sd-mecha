import torch


def convert_n_to_1(kohya_sd, kohya_keys, transpose=False):
    res = [kohya_sd[key] for key in kohya_keys]
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
