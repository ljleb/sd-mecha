import json
import multiprocessing
import os
import pathlib
import struct
from typing import Optional

import torch
import warnings


class InModelSafetensorsDict:
    def __init__(self, file_path: pathlib.Path, buffer_size):
        assert file_path.suffix == ".safetensors"
        self.default_buffer_size = buffer_size
        self.file_path = file_path
        self.file = open(file_path, 'rb')
        self.header_size, self.header = self._read_header()
        self.buffer = self.file.read(self.default_buffer_size)
        self.buffer_start_offset = 8 + self.header_size
        self.lock = multiprocessing.Lock()

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

    @property
    def is_sdxl(self):
        for key in self.keys():
            if key.startswith("conditioner"):
                return True
        return False

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
            del self.buffer
            necessary_buffer_size = max(self.default_buffer_size, length)
            self.buffer = self.file.read(necessary_buffer_size)
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


class InLoraSafetensorsDict:
    def __init__(self, file_path: pathlib.Path):
        self.safetensors_dict = InModelSafetensorsDict(file_path)
        with open(pathlib.Path(__file__).parent / "lora" / "sd15_keys.json", 'r') as f:
            self.key_map = json.load(f)

    def __del__(self):
        self.close()

    def __getitem__(self, key):
        if key.startswith("lora_"):
            raise KeyError("Direct access to 'lora_' keys is not allowed. Use target keys instead.")

        lora_key = self.key_map.get(key)
        if lora_key is None:
            raise KeyError(f"No lora key mapping found for target key: {key}")

        return self._convert_lora_to_weight(lora_key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(set(self.key_map.keys()))

    def close(self):
        self.safetensors_dict.close()

    def keys(self):
        return (
            self.key_map[key[:-len(".lora_up.weight")]]
            for key in self.safetensors_dict.keys()
            if ".lora_up.weight" in key
        )

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    @property
    def header(self):
        return self.safetensors_dict.header

    @property
    def file_path(self):
        return self.safetensors_dict.file_path

    @property
    def is_sdxl(self):
        # sdxl lora not yet supported
        return False

    def _convert_lora_to_weight(self, lora_key):
        up_weight = self.safetensors_dict[f"{lora_key}.lora_up.weight"].to(torch.float32)
        down_weight = self.safetensors_dict[f"{lora_key}.lora_down.weight"].to(torch.float32)
        alpha = self.safetensors_dict[f"{lora_key}.alpha"].to(torch.float32)
        dim = down_weight.size()[0]
        scale = alpha / dim

        if len(down_weight.size()) == 2:  # linear
            return (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):  # conv2d 1x1
            return (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
        else:  # conv2d 3x3
            return torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3) * scale


InSafetensorsDict = InModelSafetensorsDict | InLoraSafetensorsDict


class OutSafetensorsDict:
    def __init__(self, file_path: pathlib.Path, template_header: dict, buffer_size: int):
        self.file = file_path.open('wb+')
        self.lock = multiprocessing.Lock()
        self.header = {}
        self.current_offset = 0
        self.header_size = self._init_buffer(template_header)
        self.buffer_size = buffer_size
        self.buffer_used = 0
        self.buffer = bytearray(buffer_size)

    def __del__(self):
        self.file.close()

    def __setitem__(self, key: str, tensor: torch.Tensor):
        if key in self.header:
            raise ValueError("key already exists")

        tensor_bytes = tensor.cpu().numpy().tobytes()
        tensor_size = len(tensor_bytes)

        with self.lock:
            if tensor_size > len(self.buffer) - self.buffer_used:
                self._flush_buffer(next_tensor_size=tensor_size)

            self.buffer[self.buffer_used:self.buffer_used + tensor_size] = tensor_bytes
            offset = self.current_offset
            self.current_offset += tensor_size
            self.buffer_used += tensor_size

        self.header[key] = {
            "dtype": DTYPE_REVERSE_MAPPING[tensor.dtype][0],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(tensor_bytes)]
        }

    def _flush_buffer(self, next_tensor_size: Optional[int] = None):
        if self.buffer_used > 0:
            self.file.write(self.buffer[:self.buffer_used])
            self.buffer_used = 0  # Reset used buffer size

        if next_tensor_size is not None:
            # Adjust buffer size if the next tensor is larger than the default size
            required_buffer_size = max(self.buffer_size, next_tensor_size)
            if required_buffer_size != len(self.buffer):
                self.buffer = bytearray(required_buffer_size)

    def __len__(self):
        return len(self.header)

    def _init_buffer(self, template_header) -> int:
        max_digits = max(
            len(str(offset))
            for tensor_info in template_header.values()
            for offset in tensor_info.get('data_offsets', [0, 0])
        )
        dummy_number = 10 ** max(max_digits, 10) - 1

        dummy_header = {
            key: {**value, "data_offsets": [dummy_number, dummy_number]}
            for key, value in template_header.items()
        }
        header_json = json.dumps(dummy_header, separators=(',', ':')).encode('utf-8')
        header_size = len(header_json)
        self.file.seek(0)
        self.file.write(struct.pack('<Q', header_size))
        self.file.seek(header_size, os.SEEK_CUR)  # Reserve space for the header
        return header_size

    def finalize(self):
        with self.lock:
            # I'm tempted to flush all tensors in case this was the result of a very long computation
            # idk if it crashes because the header doesn't fit, then maybe there's a way for you to guess the start of each tensor...
            # good luck!
            self._flush_buffer()

            header_json = json.dumps(self.header, separators=(',', ':')).encode('utf-8')
            overhead = self.header_size - len(header_json)
            if overhead < 0:
                raise ValueError("Safetensors header does not fit into preallocated space.")

            self.file.seek(8)
            self.file.write(header_json)
            self.file.write(b' ' * overhead)
            self.file.flush()


DTYPE_MAPPING = {
    'F16': (torch.float16, 2),
    'F32': (torch.float32, 4),
    'F64': (torch.float64, 8),
    'I8': (torch.int8, 1),
    'I16': (torch.int16, 2),
    'I32': (torch.int32, 4),
    'I64': (torch.int64, 8),
}


DTYPE_REVERSE_MAPPING = {v: (k, b) for k, (v, b) in DTYPE_MAPPING.items()}
