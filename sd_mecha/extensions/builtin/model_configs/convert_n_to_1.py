import torch


def convert_n_to_1(kohya_sd, kohya_keys, transpose=False):
    res = [kohya_sd[key] for key in kohya_keys]
    if len(res) > 1:
        res = torch.vstack(res)
    else:
        res = res[0]

    if transpose:
        res = res.T

    return res
