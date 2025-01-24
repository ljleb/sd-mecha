import re
from typing import TypeVar
from sd_mecha.extensions import merge_space
from sd_mecha.extensions.merge_method import Return, Parameter, StateDict, make_recipe
from sd_mecha.extensions.merge_space import MergeSpaceSymbol


re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12


T = TypeVar("T")
SameMergeSpace = MergeSpaceSymbol(*merge_space.get_all())


# srcs, in order of priority when disagreements occur:
# - https://github.com/hako-mikan/sd-webui-supermerger/blob/f14b3e5d0be9c510d199cca502c4148160f901bb/scripts/mergers/mergers.py#L1376
# - https://github.com/s1dlx/meh/blob/04af2c8d63744fb6c02d35d328a2c84380cca444/sd_meh/merge.py#L360
# - https://github.com/vladmandic/automatic/blob/e22d0789bddd3894364b0d59a4c9b3e456e89079/modules/merging/merge_utils.py#L64
@make_recipe(is_conversion=True)
def convert_sdxl_blocks_to_sgm(
    blocks: Parameter(StateDict[T], SameMergeSpace, "sdxl_blocks-supermerger"),
    **kwargs,
) -> Return(T, SameMergeSpace, "sdxl-sgm"):
    sgm_key = kwargs["key"]

    block_key = "BASE"
    if sgm_key.startswith("model.diffusion_model."):
        block_key = "OUT08"
        if ".time_embed" in sgm_key or ".label_emb" in sgm_key:
            block_key = "BASE"  # before input blocks
        elif ".out." in sgm_key:
            block_key = "OUT08"  # after output blocks
        elif m := re_inp.search(sgm_key):
            block_key = f"IN{int(m.groups(1)[0]):02}"
        elif re_mid.search(sgm_key):
            block_key = "M00"
        elif m := re_out.search(sgm_key):
            block_key = f"OUT{int(m.groups(1)[0]):02}"
    elif sgm_key.startswith("first_stage_model."):
        block_key = "VAE"

    return blocks[block_key]
