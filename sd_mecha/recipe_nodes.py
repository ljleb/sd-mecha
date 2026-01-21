import abc
import dataclasses
import functools
import operator
import pathlib
from contextlib import contextmanager
from inspect import BoundArguments
import torch
from .extensions import model_configs, merge_methods, merge_spaces, model_dirs
from .extensions.merge_spaces import MergeSpace
from typing import Any, Callable, cast, List, Optional, Dict, Set, Tuple, Union
from .keys_map import KeyMap
from .streaming import SafetensorsMapping
from .typing_ import is_instance


class RecipeNode(abc.ABC):
    def __init__(
        self,
        model_config: Optional[model_configs.ModelConfig | str],
        merge_space: Optional[MergeSpace | str],
    ):
        if isinstance(model_config, str):
            model_config = model_configs.resolve(model_config)
        if isinstance(merge_space, str):
            merge_space = merge_spaces.resolve(merge_space)

        self.__model_config = model_config
        self.__merge_space = merge_space

    @property
    def model_config(self) -> Optional[model_configs.ModelConfig]:
        return self.__model_config

    @property
    def merge_space(self) -> Optional[MergeSpace]:
        return self.__merge_space

    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs):
        pass

    def __contains__(self, item):
        return self.accept(ContainsVisitor(item))

    def __iter__(self):
        yield from self.accept(IterVisitor())

    def __add__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return merge_methods.resolve("add_difference")(self, other)

    def __mul__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return merge_methods.resolve("scale")(self, other)

    def __radd__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return other + self

    def __sub__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return merge_methods.resolve("subtract")(self, other)

    def __rsub__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return other - self

    def __or__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return merge_methods.resolve("fallback")(self, other)

    def __ror__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = merge_methods.value_to_node(other)
        return other | self

    def to(self, *, device: Optional[Union[str, torch.device, "RecipeNode"]] = None, dtype: Optional[Union[str, torch.dtype, "RecipeNode"]] = None):
        if isinstance(device, torch.device):
            device = str(device)
        if isinstance(dtype, torch.dtype):
            from sd_mecha.extensions.builtin.merge_methods import cast_dtype_map_reversed
            dtype = cast_dtype_map_reversed[dtype]
        return merge_methods.resolve("cast")(self, device=device, dtype=dtype)


PythonLiteralValue = Optional[str | int | float | bool]
NonDictLiteralValue = PythonLiteralValue | torch.Tensor
DictLiteralValue = Dict[str, NonDictLiteralValue | RecipeNode]
LiteralValue = NonDictLiteralValue | DictLiteralValue
RecipeNodeOrValue = RecipeNode | LiteralValue | pathlib.Path


class LiteralRecipeNode(RecipeNode):
    def __init__(
        self,
        value_dict: DictLiteralValue,
        model_config: Optional[str | model_configs.ModelConfig] = None,
        merge_space: Optional[str | merge_spaces.MergeSpace] = None,
    ):
        if not isinstance(value_dict, dict):
            raise TypeError(f"value_dict is type {type(value_dict)} but it should be {DictLiteralValue}.")
        for k, v in value_dict.items():
            if not is_instance(v, NonDictLiteralValue | RecipeNode):
                raise TypeError(f"key {k} of value_dict is type {type(v)} but it should be {NonDictLiteralValue | RecipeNode}.")

        super().__init__(model_config, merge_space)
        self.value_dict = value_dict
        self.__hash = None
        self.__equal_solved = {id(self)}

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_literal(self, *args, **kwargs)

    def __eq__(self, other):
        if id(other) in self.__equal_solved:
            return True

        equal = isinstance(other, LiteralRecipeNode) and (
            self.value_dict == other.value_dict and
            self.model_config == other.model_config and
            self.merge_space == other.merge_space
        )

        if equal:
            self.__equal_solved.add(id(other))
            other.__equal_past = self.__equal_solved
        return equal

    def __hash__(self):
        if self.__hash is None:
            value_hash = functools.reduce(operator.xor, (hash(t) for t in self.value_dict.items()), -1)
            self.__hash = value_hash ^ hash(getattr(self.model_config, "identifier", None)) ^ hash(getattr(self.merge_space, "identifier", None))
        return self.__hash

    def __repr__(self) -> str:
        keys = list(self.value_dict.keys())
        keys_disp = keys[:10]
        more = len(keys) - len(keys_disp)
        keys_part = f"keys={keys_disp}" + (f" (+{more} more)" if more > 0 else "")

        cfg = getattr(self.model_config, "identifier", None)
        ms = getattr(self.merge_space, "identifier", None)

        extras = []
        if cfg is not None:
            extras.append(f"model_config={cfg}")
        if ms is not None:
            extras.append(f"merge_space={ms}")

        tail = (", " + ", ".join(extras)) if extras else ""
        return f"LiteralRecipeNode({keys_part}{tail})"


class ModelRecipeNode(RecipeNode):
    def __init__(
        self,
        path: pathlib.Path,
        model_config: Optional[str | model_configs.ModelConfig] = None,
        merge_space: Optional[str | MergeSpace] = None,
    ):
        super().__init__(model_config, merge_space)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"The type of 'path' must be Path, not {type(path).__name__}")

        self.path = path
        self.__hash = None

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_model(self, *args, **kwargs)

    @property
    @abc.abstractmethod
    def state_dict(self) -> SafetensorsMapping:
        ...

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        ...

    def __eq__(self, other):
        return isinstance(other, ModelRecipeNode) and (
            self.path == other.path and
            self.model_config == other.model_config and
            self.merge_space == other.merge_space
        )

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(self.path) ^ hash(getattr(self.model_config, "identifier", None)) ^ hash(getattr(self.merge_space, "identifier", None))
        return self.__hash

    def __repr__(self) -> str:
        cfg = getattr(self.model_config, "identifier", None)
        ms = getattr(self.merge_space, "identifier", self.merge_space)
        parts = [f"path={self.path}", f"merge_space={ms}"]
        if cfg is not None:
            parts.append(f"model_config={cfg}")
        return f"ModelRecipeNode({', '.join(parts)})"


class ClosedModelRecipeNode(ModelRecipeNode):
    @property
    def state_dict(self) -> SafetensorsMapping:
        raise RuntimeError("Model file is not open.")

    @property
    def is_open(self) -> bool:
        return False


class OpenModelRecipeNode(ModelRecipeNode):
    def __init__(
        self,
        state_dict: SafetensorsMapping,
        path: pathlib.Path,
        model_config: Optional[str | model_configs.ModelConfig] = None,
        merge_space: Optional[str | MergeSpace] = None,
    ):
        super().__init__(path, model_config, merge_space)
        self._state_dict = state_dict

    @property
    def state_dict(self) -> SafetensorsMapping:
        return self._state_dict

    @property
    def is_open(self) -> bool:
        return True


class MergeRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method: "merge_methods.MergeMethod",
        bound_args: BoundArguments,
        model_config: Optional[str | model_configs.ModelConfig] = None,
        merge_space: Optional[str | MergeSpace] = None,
    ):
        super().__init__(model_config, merge_space)
        self.merge_method = merge_method
        self.bound_args = bound_args
        self.__hash = None
        self.__equal_solved = {id(self)}
        self.__key_map = None

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_merge(self, *args, **kwargs)

    def key_map(self) -> KeyMap:
        if self.model_config is None:
            raise RuntimeError("Cannot call key_map() on a non-finalized merge node.")

        if self.__key_map is None:
            args_configs = [v.model_config for v in self.bound_args.args]
            kwargs_configs = {k: v.model_config for k, v in self.bound_args.kwargs.items()}
            self.__key_map = self.merge_method.key_map(args_configs, kwargs_configs, self.model_config)
        return self.__key_map

    def __eq__(self, other):
        if id(other) in self.__equal_solved:
            return True

        equal = isinstance(other, MergeRecipeNode) and (
            self.merge_method == other.merge_method and
            self.bound_args.arguments == other.bound_args.arguments
        )

        if equal:
            self.__equal_solved.add(id(other))
            other.__equal_past = self.__equal_solved
        return equal

    def __hash__(self):
        if self.__hash is None:
            args_hash = functools.reduce(operator.xor, (hash(t) for t in self.bound_args.arguments.items()), -1)
            self.__hash = hash(self.merge_method) ^ args_hash ^ hash(getattr(self.model_config, "identifier", None)) ^ hash(getattr(self.merge_space, "identifier", None))
        return self.__hash

    def __repr__(self) -> str:
        argc = len(self.bound_args.args)
        kwc = len(self.bound_args.kwargs)
        inputs = f"{argc} args, {kwc} kwargs"
        return f"MergeRecipeNode(method={self.merge_method.identifier}, inputs={inputs})"


class RecipeVisitor(abc.ABC):
    @abc.abstractmethod
    def visit_literal(self, node: LiteralRecipeNode):
        pass

    @abc.abstractmethod
    def visit_model(self, node: ModelRecipeNode):
        pass

    @abc.abstractmethod
    def visit_merge(self, node: MergeRecipeNode):
        pass


class ModelDepthRecipeVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        return max(
            child.accept(self) if isinstance(child, RecipeNode) else 0
            for child in node.value_dict.values()
        )

    def visit_model(self, _node: ModelRecipeNode):
        return 1

    def visit_merge(self, node: MergeRecipeNode):
        return max(
            child.accept(self)
            for child in node.bound_args.arguments.values()
        ) + 1


@dataclasses.dataclass
class ModelsCountVisitor(RecipeVisitor):
    seen: Set[pathlib.Path] = dataclasses.field(default_factory=set)

    def visit_literal(self, node: LiteralRecipeNode) -> int:
        return sum(
            v.accept(self)
            for v in set(v for v in node.value_dict.values() if isinstance(v, RecipeNode))
        )

    def visit_model(self, node: ModelRecipeNode) -> int:
        node_path = model_dirs.normalize_path(node.path)
        seen = node_path in self.seen
        self.seen.add(node_path)
        return int(not seen)

    def visit_merge(self, node: MergeRecipeNode) -> int:
        return sum(
            child.accept(self)
            for child in node.bound_args.arguments.values()
        )


@dataclasses.dataclass
class ContainsVisitor(RecipeVisitor):
    item: RecipeNode

    def visit_literal(self, node: LiteralRecipeNode):
        return node == self.item or any(
            v.accept(self)
            for v in set(v for v in node.value_dict.values() if isinstance(v, RecipeNode))
        )

    def visit_model(self, node: ModelRecipeNode):
        return node == self.item

    def visit_merge(self, node: MergeRecipeNode):
        return node == self.item or any(
            v.accept(self)
            for v in set(node.bound_args.arguments.values())
        )


@dataclasses.dataclass
class IterVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        for v in node.value_dict.values():
            if isinstance(v, RecipeNode):
                yield from v.accept(self)
        yield node

    def visit_model(self, node: ModelRecipeNode):
        yield node

    def visit_merge(self, node: MergeRecipeNode):
        for v in node.bound_args.arguments.values():
            yield from v.accept(self)
        yield node


@dataclasses.dataclass
class VisitorContext:
    stack: List[str] = dataclasses.field(default_factory=list)

    @contextmanager
    def frame(self, label: str):
        self.stack.append(label)
        try:
            yield
        finally:
            self.stack.pop()

    def trace(self) -> Tuple[str, ...]:
        return tuple(self.stack)

    def format_trace(self) -> str:
        if not self.stack:
            return "  (root)"
        return "\n".join(f"  - {s}" for s in self.stack)


@dataclasses.dataclass
class TracingRecipeVisitor(RecipeVisitor, abc.ABC):
    """
    A RecipeVisitor base that:
      - pushes a trace frame for each visited node
      - records ancestry per node in self.node_traces
      - catches exceptions and rethrows with appended trace (preserving __cause__)

    Subclasses can keep defining visit_literal/visit_model/visit_merge as usual.
    Just inherit from TracingRecipeVisitor instead of RecipeVisitor.
    """

    def __post_init__(self):
        self.ctx: VisitorContext = VisitorContext()
        self.node_traces: Dict["RecipeNode", Tuple[str, ...]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        for name in ("visit_literal", "visit_model", "visit_merge"):
            orig = cls.__dict__.get(name)
            if orig is None:
                continue
            setattr(cls, name, cls._wrap_visit(cast(Callable[..., Any], orig)))

    @staticmethod
    def _wrap_visit(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: "TracingRecipeVisitor", node: Any, *args: Any, **kwargs: Any):
            label = f"{node.__class__.__name__}: {node!r}"
            with self.ctx.frame(label):
                self.node_traces[node] = self.ctx.trace()

                try:
                    return fn(self, node, *args, **kwargs)
                except Exception as e:
                    raise self._augment_exception(e) from e

        wrapped.__name__ = getattr(fn, "__name__", "wrapped_visit")
        wrapped.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        return wrapped

    def _augment_exception(self, e: Exception) -> Exception:
        msg = str(e)
        if "\nTrace:\n" in msg:
            return e

        trace = self.ctx.format_trace()
        augmented = f"{msg}\nTrace:\n{trace}"

        etype = type(e)
        try:
            return etype(augmented)
        except Exception:
            return RuntimeError(augmented)
