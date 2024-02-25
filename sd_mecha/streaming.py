import contextlib
import dataclasses
import json
import pathlib
import struct
import threading
from typing import Optional, Set
import torch
import warnings

from torch._subclasses import FakeTensorMode
from tqdm import tqdm


class InSafetensorsDict:
    def __init__(self, file_path: pathlib.Path, buffer_size):
        if not file_path.suffix == ".safetensors":
            raise ValueError(f"Model type not supported: {file_path} (only safetensors are supported)")

        self.default_buffer_size = buffer_size
        self.file_path = file_path
        self.file = open(file_path, mode='rb', buffering=0)
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

    def __setitem__(self, key: str, tensor: torch.Tensor):
        tid = threading.current_thread().ident
        if tid not in self.thread_states:
            self.thread_states[tid] = OutSafetensorsDictThreadState(bytearray(self.minimum_buffer_size))

        state = self.thread_states[tid]

        tensor_bytes = tensor.cpu().numpy().tobytes()
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
        fake_mode = FakeTensorMode()

        def _create_fake_tensor(*shape, dtype):
            return fake_mode.from_tensor(torch.empty(shape, dtype=dtype))

        with fake_mode:
            fake_state_dict = {
                k: _create_fake_tensor(*h["shape"], dtype=DTYPE_MAPPING[h["dtype"]][0])
                for k, h in template_header.items()
                if k != "__metadata__"
            }
            fake_state_dict = dict(sorted(
                fake_state_dict.items(),
                key=lambda x: x[1].numel() * x[1].element_size(),
                reverse=True,
            ))

            data_offset = 0
            dummy_header = {
                k: {
                    "dtype": DTYPE_REVERSE_MAPPING[save_dtype][0] if k in keys_to_merge else DTYPE_REVERSE_MAPPING[t.dtype][0],
                    "shape": list(t.shape),
                    "data_offsets": [data_offset, (data_offset := data_offset + t.numel() * t.element_size())],
                }
                for k, t in fake_state_dict.items()
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


DTYPE_MAPPING = {
    'F64': (torch.float64, 8),
    'F32': (torch.float32, 4),
    'F16': (torch.float16, 2),
    'BF16': (torch.bfloat16, 2),
    'I8': (torch.int8, 1),
    'I64': (torch.int64, 8),
    'I32': (torch.int32, 4),
    'I16': (torch.int16, 2),
}


DTYPE_REVERSE_MAPPING = {v: (k, b) for k, (v, b) in DTYPE_MAPPING.items()}
