import pathlib
from typing import Optional

from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha import ast_nodes


def merge_and_save(
    merge_tree: ast_nodes.MergeNode,
    base_dir,
    output_path,
):
    scheduler = MergeScheduler(base_dir=base_dir)
    scheduler.merge_and_save(merge_tree, output_path=output_path)


def weighted_sum(
    a, b, *,
    alpha: float = 0.5,
    rebasin_iters: Optional[int] = None,
    precision: int = 16,
    prune: bool = False,
    threads: int = 1,
    device: str = "cpu",
    work_device: Optional[str] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if isinstance(a, (str, pathlib.Path)):
        a = ast_nodes.LeafMergeNode(a, device)
    if isinstance(b, (str, pathlib.Path)):
        b = ast_nodes.LeafMergeNode(b, device)

    return ast_nodes.SymbolicMergeNode(
        merge_method="weighted_sum",
        a=a,
        b=b,
        alpha=alpha,
        rebasin_iters=rebasin_iters,
        precision=precision,
        prune=prune,
        threads=threads,
        device=device,
        work_device=work_device,
        weights_clip=clip_weights_to_ab,
    )


def add_difference(
    a, b, c, *,
    alpha: float = 1.0,
    precision: int = 16,
    prune: bool = False,
    threads: int = 1,
    device: str = "cpu",
    work_device: Optional[str] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if isinstance(a, (str, pathlib.Path)):
        a = ast_nodes.LeafMergeNode(a, device)
    if isinstance(b, (str, pathlib.Path)):
        b = ast_nodes.LeafMergeNode(b, device)
    if isinstance(c, (str, pathlib.Path)):
        c = ast_nodes.LeafMergeNode(c, device)

    return ast_nodes.SymbolicMergeNode(
        merge_method="add_difference",
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        precision=precision,
        prune=prune,
        threads=threads,
        device=device,
        work_device=work_device,
        weights_clip=clip_weights_to_ab,
    )


def rotate(
    a, b, *,
    alpha: float = 1.0,
    beta: float = 0.0,
    precision: int = 16,
    prune: bool = False,
    threads: int = 1,
    device: str = "cpu",
    work_device: Optional[str] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if isinstance(a, (str, pathlib.Path)):
        a = ast_nodes.LeafMergeNode(a, device)
    if isinstance(b, (str, pathlib.Path)):
        b = ast_nodes.LeafMergeNode(b, device)

    return ast_nodes.SymbolicMergeNode(
        merge_method="add_difference",
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
        precision=precision,
        prune=prune,
        threads=threads,
        device=device,
        work_device=work_device,
        weights_clip=clip_weights_to_ab,
    )


def add_perpendicular(
    a, b, c, *,
    alpha: float = 1.0,
    rebasin_iters: Optional[int] = None,
    precision: int = 16,
    prune: bool = False,
    threads: int = 1,
    device: str = "cpu",
    work_device: Optional[str] = None,
    clip_weights_to_ab: bool = False,
) -> ast_nodes.MergeNode:
    if isinstance(a, (str, pathlib.Path)):
        a = ast_nodes.LeafMergeNode(a, device)
    if isinstance(b, (str, pathlib.Path)):
        b = ast_nodes.LeafMergeNode(b, device)
    if isinstance(c, (str, pathlib.Path)):
        c = ast_nodes.LeafMergeNode(c, device)

    return ast_nodes.SymbolicMergeNode(
        merge_method="add_perpendicular",
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        rebasin_iters=rebasin_iters,
        precision=precision,
        prune=prune,
        threads=threads,
        device=device,
        work_device=work_device,
        weights_clip=clip_weights_to_ab,
    )


def clip(model, a, b, device: str = None) -> ast_nodes.MergeNode:
    if isinstance(a, (str, pathlib.Path)):
        a = ast_nodes.LeafMergeNode(a, device)
    if isinstance(b, (str, pathlib.Path)):
        b = ast_nodes.LeafMergeNode(b, device)

    return ast_nodes.ClipMergeNode(model, a, b, device)
