import contextlib
import ctypes
import dataclasses
import functools
import json
import operator
import pathlib
import struct
import sys
import threading
import numpy
import torch
import warnings
from typing import Optional, Set
from tqdm import tqdm


class DiffusersInSafetensorsDict:
    def __init__(self, dir_path: pathlib.Path, model_arch, buffer_size: int):
        self.file_path = dir_path
        self.dicts = {
            component: InSafetensorsDict(find_best_safetensors_path(dir_path / component), buffer_size // len(model_arch.components))
            for component in model_arch.components
        }
        self.model_arch = model_arch

    def __del__(self):
        self.close()

    def __getitem__(self, item):
        component, key = item.split(".", maxcount=1)
        if key in self.dicts[component]:
            return self.dicts[component][key]

        raise KeyError(item)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(len(d) for d in self.dicts)

    def close(self):
        for d in self.dicts:
            d.close()

    def keys(self):
        for component, d in self.dicts.items():
            for key in d:
                if key != "__metadata__":
                    yield f"{component}.{key}"

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]


def find_best_safetensors_path(dir_path: pathlib.Path) -> pathlib.Path:
    best_file = None
    for file in dir_path.iterdir():
        if file.is_file() and "model" in file.name and file.suffix == ".safetensors":
            if best_file is None or "fp16" in str(best_file.name):
                best_file = file

    if best_file is None:
        raise RuntimeError(f"could not find safetensors model file under {dir_path}")

    return best_file


class DiffusersOutSafetensorsDict:
    def __init__(
        self,
        dir_path: pathlib.Path,
        model_arch,
        template_header: dict,
        keys_to_merge: Set[str],
        mecha_recipe: str,
        minimum_buffer_size: int,
        save_dtype: torch.dtype,
    ):
        dir_path.mkdir(exist_ok=True)
        for component in model_arch.components:
            (dir_path / component).mkdir(exist_ok=True)

        self.file_path = dir_path
        self.dicts = {
            component: OutSafetensorsDict(
                dir_path / component / "model.safetensors",
                {k.split(".", maxsplit=1)[1]: v for k, v in template_header.items() if k.startswith(component + ".")},
                {k.split(".", maxsplit=1)[1] for k in keys_to_merge if k.startswith(component + ".")},
                mecha_recipe,
                minimum_buffer_size // len(model_arch.components),
                save_dtype,
            )
            for component in model_arch.components
        }
        self.model_arch = model_arch

    def __del__(self):
        self.close()

    def __setitem__(self, item, value):
        component, key = item.split(".", maxcount=1)
        self.dicts[component][key] = value

    def __len__(self):
        return sum(len(d) for d in self.dicts)

    def close(self):
        for d in self.dicts:
            d.close()


class InSafetensorsDict:
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

    def __getitem__(self, key):
        if key not in self.header or key == "__metadata__":
            raise KeyError(key)
        return self._load_tensor(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.header)

    def close(self):
        self.file.close()
        self.buffer = None
        self.header = None

    def keys(self):
        return (
            key
            for key in self.header.keys()
            if key != "__metadata__"
        )

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def _read_header(self):
        header_size_bytes = self.file.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_json = self.file.read(header_size).decode('utf-8').strip()
        header = json.loads(header_json)

        # sort by memory order to reduce seek time
        sorted_header = dict(sorted(header.items(), key=lambda item: item[1].get('data_offsets', [0])[0]))
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
        absolute_start_pos = 8 + self.header_size + offsets[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.lock:
                self._ensure_buffer(absolute_start_pos, total_bytes)
                buffer_offset = absolute_start_pos - self.buffer_start_offset
                return torch.frombuffer(self.buffer, count=total_bytes // dtype_bytes, offset=buffer_offset, dtype=dtype).reshape(shape)


@dataclasses.dataclass
class OutSafetensorsDictThreadState:
    buffer: bytearray
    memory_used: int = dataclasses.field(default=0)
    sub_header: dict = dataclasses.field(default_factory=dict)


class OutSafetensorsDict:
    def __init__(
        self,
        file_path: pathlib.Path,
        template_header: dict,
        keys_to_merge: Set[str],
        mecha_recipe: str,
        minimum_buffer_size: int,
        save_dtype: torch.dtype,
    ):
        self.thread_states = {}
        self.lock = threading.Lock()

        self.header = {
            "__metadata__": {"mecha_recipe": mecha_recipe}
        }
        self.file = file_path.open("wb", buffering=0)
        self.file_path = file_path
        self.flushed_size = 0
        self.minimum_buffer_size = minimum_buffer_size

        self.max_header_size = self._init_buffer(template_header, keys_to_merge, save_dtype)

    def __del__(self):
        self.file.close()
        self.thread_states = None
        self.header = None

    def __setitem__(self, key: str, tensor: torch.Tensor):
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

    def __len__(self):
        return len(self.header)

    def _init_buffer(self, template_header: dict, keys_to_merge: Set[str], save_dtype: torch.dtype) -> int:
        def get_dtype(k):
            return DTYPE_REVERSE_MAPPING[save_dtype][0] if k in keys_to_merge else template_header[k]["dtype"]

        def get_dtype_size(k):
            return DTYPE_REVERSE_MAPPING[save_dtype][1] if k in keys_to_merge else DTYPE_MAPPING[template_header[k]["dtype"]][1]

        def get_width(k):
            return functools.reduce(operator.mul, template_header[k]["shape"], 1) * get_dtype_size(k)

        keys = sorted(
            list(k for k in template_header.keys() if k != "__metadata__"),
            key=get_width,
            reverse=True,  # simulate worst case: maximize space taken by order
        )
        data_offset = 0
        dummy_header = {
            k: {
                "dtype": get_dtype(k),
                "shape": template_header[k]["shape"],
                "data_offsets": [data_offset, (data_offset := data_offset + get_width(k))],
            }
            for k in keys
        }

        dummy_header["__metadata__"] = self.header["__metadata__"]
        header_json = json.dumps(dummy_header, separators=(',', ':')).encode('utf-8')
        max_header_size = len(header_json)
        self.file.seek(8 + max_header_size)  # Reserve space for the header
        return max_header_size

    def _flush_buffer(self, state: OutSafetensorsDictThreadState, next_tensor_size: Optional[int] = None, close: bool = False):
        if not close:
            lock_obj = self.lock
        else:
            lock_obj = contextlib.nullcontext()

        with lock_obj:
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
# this function is Apache 2.0: https://github.com/huggingface/safetensors/blob/main/LICENSE
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
    'F64': (torch.float64, 8),
    'F32': (torch.float32, 4),
    'F16': (torch.float16, 2),
    'BF16': (torch.bfloat16, 2),
    'I8': (torch.int8, 1),
    'I64': (torch.int64, 8),
    'I32': (torch.int32, 4),
    'I16': (torch.int16, 2),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
}


DTYPE_REVERSE_MAPPING = {v: (k, b) for k, (v, b) in DTYPE_MAPPING.items()}
