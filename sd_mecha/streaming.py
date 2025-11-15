import abc
import contextlib
import ctypes
import dataclasses
import json
import pathlib
import struct
import sys
import threading
import numpy
import torch
import warnings
from collections import OrderedDict
from typing import Optional, Mapping, Iterator, Iterable, Tuple
from tqdm import tqdm
from .typing_ import WriteOnlyMapping


@dataclasses.dataclass
class TensorMetadata:
    shape: Optional[torch.Size]
    dtype: Optional[torch.dtype]

    def __post_init__(self):
        if isinstance(self.shape, list):
            self.shape = torch.Size(self.shape)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

    def safetensors_header_value(self, data_offset: int):
        if self.shape is None:
            raise RuntimeError("invalid operation: metadata doesn't have shape")

        if self.dtype is None:
            raise RuntimeError("invalid operation: metadata doesn't have dtype")

        return {
            "shape": list(self.shape),
            "dtype": DTYPE_REVERSE_MAPPING[self.dtype][0],
            "data_offsets": [data_offset, data_offset + self.get_byte_size()]
        }

    def get_byte_size(self) -> int:
        return self.numel() * self.get_dtype_size()

    def get_dtype_size(self) -> int:
        if self.dtype is None:
            raise RuntimeError("invalid operation: metadata doesn't have dtype")

        return DTYPE_REVERSE_MAPPING[self.dtype][1]

    def numel(self) -> int:
        if self.shape is None:
            raise RuntimeError("invalid operation: metadata doesn't have shape")

        return self.shape.numel()


class SafetensorsMapping(Mapping[str, torch.Tensor], abc.ABC):
    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def metadata(self) -> Iterable[Tuple[str, TensorMetadata]]:
        ...

    @abc.abstractmethod
    def values(self) -> Iterable[torch.Tensor]:
        ...

    @abc.abstractmethod
    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        ...


class InSafetensorsDict(SafetensorsMapping):
    def __init__(self, file_path: pathlib.Path, buffer_size):
        if not file_path.suffix == ".safetensors":
            raise ValueError(f"Model type not supported: {file_path} (only safetensors are supported)")

        self.default_buffer_size = buffer_size
        self.file = open(file_path, mode='rb', buffering=0)
        self.file_path = file_path
        self.header_size, self.header = self._read_header()
        self.buffer = bytearray()
        self.buffer_start_offset = 8 + self.header_size
        self.lock = threading.Lock()

    def __del__(self):
        self.close()

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self.header or key == "__metadata__":
            raise StateDictKeyError(key)
        return self._load_tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.header) - int("__metadata__" in self.header)

    def close(self):
        if getattr(self, "file", None) is not None:
            self.file.close()
        self.file = None
        self.buffer = None
        self.header = None

    def keys(self) -> Iterable[str]:
        return (
            key
            for key in self.header.keys()
            if key != "__metadata__"
        )

    def metadata(self) -> Iterable[Tuple[str, TensorMetadata]]:
        for key in self.keys():
            yield key, TensorMetadata(self.header[key]["shape"],  DTYPE_MAPPING[self.header[key]["dtype"]][0])

    def values(self) -> Iterable[torch.Tensor]:
        for key in self.keys():
            yield self[key]

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        for key in self.keys():
            yield key, self[key]

    def _read_header(self):
        header_size_bytes = self.file.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_json = self.file.read(header_size).decode('utf-8').strip()
        header = json.loads(header_json)

        # sort by memory order to reduce seek time
        sorted_header = OrderedDict(sorted(header.items(), key=lambda item: item[1].get('data_offsets', [0])[0]))
        return header_size, sorted_header

    def _ensure_buffer(self, start_pos, length):
        if start_pos < self.buffer_start_offset or start_pos + length > self.buffer_start_offset + len(self.buffer):
            self.file.seek(start_pos)
            necessary_buffer_size = max(self.default_buffer_size, length)
            if len(self.buffer) < necessary_buffer_size:
                self.buffer = bytearray(necessary_buffer_size)
            else:
                self.buffer = self.buffer[:necessary_buffer_size]

            self.file.readinto(self.buffer)
            self.buffer_start_offset = start_pos

    def _load_tensor(self, tensor_name):
        tensor_info = self.header[tensor_name]
        offsets = tensor_info['data_offsets']
        dtype, dtype_bytes = DTYPE_MAPPING[tensor_info['dtype']]
        shape = tensor_info['shape']
        total_bytes = offsets[1] - offsets[0]
        if total_bytes == 0:
            return torch.tensor([], dtype=dtype).reshape(shape)

        absolute_start_pos = 8 + self.header_size + offsets[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.lock:
                self._ensure_buffer(absolute_start_pos, total_bytes)
                buffer_offset = absolute_start_pos - self.buffer_start_offset
                return torch.frombuffer(self.buffer, count=total_bytes // dtype_bytes, offset=buffer_offset, dtype=dtype).reshape(shape)


class StateDictKeyError(KeyError):
    """
    Exception raised when a requested key is missing from a streamed or in-memory state dict.

    It behaves like a normal `KeyError`, but is specialized for reporting missing keys
    within streaming merges or recipes.
    """


@dataclasses.dataclass
class OutSafetensorsDictThreadState:
    buffer: bytearray
    memory_used: int = dataclasses.field(default=0)
    sub_header: dict = dataclasses.field(default_factory=dict)


class OutSafetensorsDict(WriteOnlyMapping[str, torch.Tensor]):
    def __init__(
        self,
        file_path: pathlib.Path,
        header: Mapping[str, TensorMetadata],
        mecha_recipe: Optional[str],
        minimum_buffer_size: int,
    ):
        self.thread_states = {}
        self.lock = threading.Lock()

        self.header = {
            "__metadata__": {"mecha_recipe": mecha_recipe} if mecha_recipe is not None else {}
        }
        self.file = file_path.open("wb", buffering=0)
        self.file_path = file_path
        self.flushed_size = 0
        self.minimum_buffer_size = minimum_buffer_size

        self.max_header_size = self._init_buffer(header)

    def __del__(self):
        self.file.close()
        self.thread_states = None
        self.header = None

    def __setitem__(self, key: str, tensor: torch.Tensor) -> None:
        tid = threading.current_thread().ident
        if tid not in self.thread_states:
            self.thread_states[tid] = OutSafetensorsDictThreadState(bytearray(self.minimum_buffer_size))

        state = self.thread_states[tid]

        tensor_bytes = tensor_to_bytes(tensor)
        tensor_size = len(tensor_bytes)

        if tensor_size > len(state.buffer) - state.memory_used:
            self._flush_buffer(state, next_tensor_size=tensor_size)

        local_offset = state.memory_used
        state.buffer[state.memory_used:state.memory_used + tensor_size] = tensor_bytes
        state.memory_used += tensor_size

        state.sub_header[key] = {
            "dtype": DTYPE_REVERSE_MAPPING[tensor.dtype][0],
            "shape": list(tensor.shape),
            "data_offsets": [local_offset, local_offset + tensor_size]
        }

    def __len__(self) -> int:
        return len(self.header)

    def _init_buffer(self, header: Mapping[str, TensorMetadata]) -> int:
        trimmed_header = {
            k: v for k, v in header.items() if v.shape is not None and v.dtype is not None
        }
        worst_case_header = OrderedDict(sorted(
            trimmed_header.items(),
            key=lambda item: item[1].get_byte_size(),
            reverse=True,  # simulate worst case: maximize space taken by order
        ))

        data_offset = 0
        dummy_safetensors_header = OrderedDict(self.header)
        for k, v in worst_case_header.items():
            dummy_safetensors_header[k] = v.safetensors_header_value(data_offset)
            data_offset += v.get_byte_size()

        header_json = json.dumps(dummy_safetensors_header, separators=(',', ':')).encode('utf-8')
        max_header_size = len(header_json)
        self.file.seek(8 + max_header_size)  # Reserve space for the header
        return max_header_size

    def _flush_buffer(self, state: OutSafetensorsDictThreadState, next_tensor_size: Optional[int] = None, close: bool = False):
        if not close:
            lock = self.lock
        else:
            lock = contextlib.nullcontext()

        with lock:
            self.file.write(state.buffer[:state.memory_used])
            buffer_offset = self.flushed_size
            self.flushed_size += state.memory_used
            state.memory_used = 0

        if next_tensor_size is not None:
            required_buffer_size = max(self.minimum_buffer_size, next_tensor_size)
            if len(state.buffer) < required_buffer_size:
                state.buffer = bytearray(required_buffer_size)
            else:
                state.buffer = state.buffer[:required_buffer_size]

        global_sub_header = {
            k: {
                attr: val
                if attr != "data_offsets"
                else (val[0] + buffer_offset, val[1] + buffer_offset)
                for attr, val in v.items()
            }
            for k, v in state.sub_header.items()
        }
        self.header.update(global_sub_header)
        state.sub_header.clear()
        if close:
            state.buffer = b""

    def close(self):
        with self.lock:
            for state in self.thread_states.values():
                self._flush_buffer(state, close=True)

            header_json = json.dumps(self.header, separators=(',', ':')).encode('utf-8')
            header_size = len(header_json)
            overhead = self.max_header_size - header_size

            if overhead < 0:
                # not enough space. we have to move the entire data section by `-overhead`
                # this should never happen, but it's here just in case as a fallback
                data_offset = -overhead
                old_data_section = 8 + self.max_header_size
                old_file_end = 8 + self.max_header_size + self.flushed_size
                new_file_end = 8 + header_size + self.flushed_size
                self.file.truncate(new_file_end)

                # close and reopen the file in read-write mode
                self.file.close()
                self.file = open(self.file_path, "rb+")

                # move data in chunks from the end to avoid overwriting
                for chunk_end in tqdm(range(old_file_end, old_data_section, -self.minimum_buffer_size), desc="Reallocating data section"):
                    chunk_start = max(chunk_end - self.minimum_buffer_size, old_data_section)
                    chunk_size = chunk_end - chunk_start
                    self.file.seek(chunk_start)
                    data = self.file.read(chunk_size)

                    # calculate the new position and write the chunk
                    self.file.seek(chunk_start + data_offset)
                    self.file.write(data)

                # we made just enough space for the header
                overhead = 0

            self.file.seek(0)
            self.file.write(struct.pack('<Q', max(self.max_header_size, header_size)))
            self.file.write(header_json)
            self.file.write(b' ' * overhead)
            self.file.close()


# src: https://github.com/huggingface/safetensors/blob/aa4ad823cf71d913f283b70332d37ab45803949d/bindings/python/py_src/safetensors/torch.py#L405
# this function is Apache 2.0, see /LICENSE-safetensors.txt
def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    # assume tensor is not spare nor contiguous and on the cpu
    total_bytes = len(tensor.untyped_storage())

    ptr = tensor.data_ptr()
    if ptr == 0:
        return b""
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = numpy.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    if sys.byteorder == "big":
        NPDTYPES = {
            torch.int64: numpy.int64,
            torch.float32: numpy.float32,
            torch.int32: numpy.int32,
            # XXX: This is ok because both have the same width
            torch.bfloat16: numpy.float16,
            torch.float16: numpy.float16,
            torch.int16: numpy.int16,
            torch.uint8: numpy.uint8,
            torch.int8: numpy.int8,
            torch.bool: bool,
            torch.float64: numpy.float64,
            # XXX: This is ok because both have the same width and byteswap is a no-op anyway
            torch.float8_e4m3fn: numpy.uint8,
            torch.float8_e5m2: numpy.uint8,
        }
        npdtype = NPDTYPES[tensor.dtype]
        # Not in place as that would potentially modify a live running model
        data = data.view(npdtype).byteswap(inplace=False)
    return data.tobytes()


DTYPE_MAPPING = {
    "F64": (torch.float64, 8),
    "I64": (torch.int64, 8),
    "F32": (torch.float32, 4),
    "I32": (torch.int32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "BOOL": (torch.bool, 1),
}
for i, dtype_str in enumerate(("uint8", "uint16", "uint32", "uint64")):
    if hasattr(torch, dtype_str):
        num_bytes = 2**i
        num_bits = 8 * num_bytes
        DTYPE_MAPPING[f"U{num_bits}"] = (getattr(torch, dtype_str), num_bytes)

DTYPE_REVERSE_MAPPING = {v: (k, b) for k, (v, b) in DTYPE_MAPPING.items()}
