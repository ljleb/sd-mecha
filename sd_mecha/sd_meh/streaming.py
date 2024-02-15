import json
import os
import pathlib
import struct
import warnings

import torch


DTYPE_MAPPING = {
    'F16': torch.float16,
    'F32': torch.float32,
    'F64': torch.float64,
    'I8': torch.int8,
    'I16': torch.int16,
    'I32': torch.int32,
    'I64': torch.int64,
}


DTYPE_REVERSE_MAPPING = {v: k for k, v in DTYPE_MAPPING.items()}


class InSafetensorDict:
    def __init__(self, file_path: pathlib.Path):
        assert file_path.suffix == ".safetensors"
        self.file = open(file_path, 'rb')
        self.header_size, self.header = self._read_header()

    def __del__(self):
        self.close()

    def __getitem__(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found.")
        return self._load_tensor(key)

    def __iter__(self):
        return iter(self.header.keys())

    def __len__(self):
        return len(self.header)

    def close(self):
        self.file.close()

    def keys(self):
        return self.header.keys()

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        return zip(self.keys(), self.values())

    def _read_header(self):
        header_size_bytes = self.file.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_json = self.file.read(header_size).decode('utf-8').strip()
        header = json.loads(header_json)

        # sort by memory order to reduce seek time
        sorted_header = dict(sorted(header.items(), key=lambda item: item[1].get('data_offsets', [0])[0]))
        return header_size, sorted_header

    def _load_tensor(self, tensor_name):
        tensor_info = self.header[tensor_name]
        offsets = tensor_info['data_offsets']
        dtype = DTYPE_MAPPING[tensor_info['dtype']]
        shape = tensor_info['shape']
        total_bytes = offsets[1] - offsets[0]
        absolute_start_pos = 8 + self.header_size + offsets[0]
        self.file.seek(absolute_start_pos)
        data = self.file.read(total_bytes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.frombuffer(data, dtype=dtype).reshape(shape)


class OutSafetensorDict:
    def __init__(self, file_path: pathlib.Path, template_header: dict):
        self.file = file_path.open('wb+')
        self.header = {}
        self.current_offset = 0
        self.header_size = self._init_buffer(template_header)

    def __del__(self):
        self.file.close()

    def __setitem__(self, key: str, tensor: torch.Tensor):
        if key in self.header:
            raise ValueError("key already exists")

        tensor_bytes = tensor.cpu().numpy().tobytes()
        self.file.write(tensor_bytes)
        self.header[key] = {
            "dtype": DTYPE_REVERSE_MAPPING[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [self.current_offset, self.current_offset + len(tensor_bytes)]
        }
        self.current_offset += len(tensor_bytes)

    def __len__(self):
        return len(self.header)

    def _init_buffer(self, template_header) -> int:
        max_digits = max(
            len(str(offset))
            for tensor_info in template_header.values()
            for offset in tensor_info.get('data_offsets', [0, 0])
        )
        dummy_number = 10 ** max_digits - 1

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
        header_json = json.dumps(self.header, separators=(',', ':')).encode('utf-8')
        self.file.seek(8)
        self.file.write(header_json)
        overhead = self.header_size - len(header_json)
        if overhead < 0:
            raise ValueError("safetensors header does not fit into memory")
        self.file.write(b' ' * overhead)
