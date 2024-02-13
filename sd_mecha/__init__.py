import logging
import pathlib
import torch
from tensordict import TensorDict
from typing import Optional
from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha import ast_nodes


MergeNode = ast_nodes.MergeNode | str | pathlib.Path | TensorDict


def merge_and_save(
    merge_tree: ast_nodes.MergeNode,
    base_dir,
    output_path,
):
    scheduler = MergeScheduler(base_dir=base_dir)
    scheduler.merge_and_save(merge_tree, output_path=output_path)


def weighted_sum(
    a: MergeNode, b: MergeNode, *,
    alpha: float = 0.5,
    rebasin_iters: Optional[int] = None,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if not isinstance(a, ast_nodes.MergeNode):
        a = ast_nodes.LeafMergeNode(a)
    if not isinstance(b, ast_nodes.MergeNode):
        b = ast_nodes.LeafMergeNode(b)

    return ast_nodes.SymbolicMergeNode(
        merge_method="weighted_sum",
        a=a,
        b=b,
        alpha=alpha,
        rebasin_iters=rebasin_iters,
        threads=threads,
        device=device,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def add_difference(
    a: MergeNode, b: MergeNode, c: MergeNode, *,
    alpha: float = 0.5,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if not isinstance(a, ast_nodes.MergeNode):
        a = ast_nodes.LeafMergeNode(a)
    if not isinstance(b, ast_nodes.MergeNode):
        b = ast_nodes.LeafMergeNode(b)
    if not isinstance(c, ast_nodes.MergeNode):
        c = ast_nodes.LeafMergeNode(c)

    return ast_nodes.SymbolicMergeNode(
        merge_method="add_difference",
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        threads=threads,
        device=device,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def tensor_sum(
    a: MergeNode, b: MergeNode, *,
    width: float = 0.5,
    offset: float = 0.0,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
) -> ast_nodes.MergeNode:
    if not isinstance(a, ast_nodes.MergeNode):
        a = ast_nodes.LeafMergeNode(a)
    if not isinstance(b, ast_nodes.MergeNode):
        b = ast_nodes.LeafMergeNode(b)

    return ast_nodes.SymbolicMergeNode(
        merge_method="tensor_sum",
        a=a,
        b=b,
        alpha=width,
        beta=offset,
        threads=threads,
        device=device,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def add_perpendicular(
    a: MergeNode, b: MergeNode, c: MergeNode, *,
    alpha: float = 1.0,
    rebasin_iters: Optional[int] = None,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if not isinstance(a, ast_nodes.MergeNode):
        a = ast_nodes.LeafMergeNode(a)
    if not isinstance(b, ast_nodes.MergeNode):
        b = ast_nodes.LeafMergeNode(b)
    if not isinstance(c, ast_nodes.MergeNode):
        c = ast_nodes.LeafMergeNode(c)

    return ast_nodes.SymbolicMergeNode(
        merge_method="add_perpendicular",
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        rebasin_iters=rebasin_iters,
        threads=threads,
        device=device,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def rotate(
    a: MergeNode, b: MergeNode, *,
    alpha: float = 1.0,
    beta: float = 0.0,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if not isinstance(a, ast_nodes.MergeNode):
        a = ast_nodes.LeafMergeNode(a)
    if not isinstance(b, ast_nodes.MergeNode):
        b = ast_nodes.LeafMergeNode(b)

    return ast_nodes.SymbolicMergeNode(
        merge_method="rotate",
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
        threads=threads,
        device=device,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def clip(model: MergeNode, a: MergeNode, b: MergeNode) -> ast_nodes.MergeNode:
    if not isinstance(a, ast_nodes.MergeNode):
        a = ast_nodes.LeafMergeNode(a)
    if not isinstance(b, ast_nodes.MergeNode):
        b = ast_nodes.LeafMergeNode(b)

    return ast_nodes.ClipMergeNode(model, a, b)


def model(
    state_dict: str | pathlib.Path | TensorDict,
    device: str = None,
):
    return ast_nodes.LeafMergeNode(state_dict, device)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
