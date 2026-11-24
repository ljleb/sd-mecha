import torch


def moss_image_missing_key_handler(config, key, device, dtype):
    if key == "rope.bands":
        # OpenMOSS constructs these frequencies in float32 on the CPU.
        count = config.keys()[key].shape[0]
        exponents = torch.arange(count, device="cpu", dtype=torch.float32) / count
        return (1 / 10000.0 ** exponents).to(device=device)
    return None
