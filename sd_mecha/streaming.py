import json
import mmap
import multiprocessing
import os
import pathlib
import struct
import torch
import warnings
from collections import OrderedDict


class InModelSafetensorsDict:
    def __init__(self, file_path: pathlib.Path):
        assert file_path.suffix == ".safetensors"
        self.file_path = file_path
        self.file = open(file_path, 'rb')
        self.header_size, self.header = self._read_header()

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
    def __init__(self, file_path: pathlib.Path, template_header: dict):
        self.file = file_path.open('wb+')
        self.lock = multiprocessing.Lock()
        self.header = {}
        self.current_offset = 0
        self.header_size = self._init_buffer(template_header)

    def __del__(self):
        self.file.close()

    def __setitem__(self, key: str, tensor: torch.Tensor):
        if key in self.header:
            raise ValueError("key already exists")

        tensor_bytes = tensor.cpu().numpy().tobytes()
        with self.lock:
            offset = self.current_offset
            self.current_offset += len(tensor_bytes)
            self.file.write(tensor_bytes)

        self.header[key] = {
            "dtype": DTYPE_REVERSE_MAPPING[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(tensor_bytes)]
        }

    def __len__(self):
        return len(self.header)

    def _init_buffer(self, template_header) -> int:
        max_digits = max(
            len(str(offset))
            for tensor_info in template_header.values()
            for offset in tensor_info.get('data_offsets', [0, 0])
        )
        dummy_number = 10 ** min(max_digits, 10) - 1

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


class BufferedMMap:
    def __init__(self, file, max_pages, read_only=True):
        self.file = file
        self.file_size = os.fstat(file.fileno()).st_size
        self.max_pages = max_pages
        self.page_size = mmap.ALLOCATIONGRANULARITY * 64
        self.lock = multiprocessing.Lock()
        self.lru_cache = OrderedDict()
        self.read_only = read_only
        self._ensure_mapped(0, self.file_size)

    def __del__(self):
        self.close()

    def close(self):
        with self.lock:
            for mmap_obj in self.lru_cache.values():
                mmap_obj.close()
            self.lru_cache.clear()

    def _ensure_mapped(self, offset, length, move_to_end=True):
        page_start = offset // self.page_size
        page_end = min((offset + length) // self.page_size + 1 - page_start, self.max_pages) + page_start

        for page_number in range(page_start, page_end):
            if page_number * self.page_size not in self.lru_cache:
                self._load_page(page_number)
            elif move_to_end:
                # Move the page to the end to mark it as recently used
                self.lru_cache.move_to_end(page_number * self.page_size)

    def _load_page(self, page_number):
        start_offset = page_number * self.page_size
        end_offset = min(start_offset + self.page_size, self.file_size)

        if len(self.lru_cache) >= self.max_pages:
            self._unload_oldest_page()

        new_map = mmap.mmap(
            self.file.fileno(),
            end_offset - start_offset,
            access=mmap.ACCESS_READ if self.read_only else mmap.ACCESS_WRITE,
            offset=start_offset
        )
        self.lru_cache[start_offset] = new_map
        # No need to explicitly track offsets as keys in `self.mmaps` serve this purpose

    def _unload_oldest_page(self):
        # Unload the least recently used page
        oldest_offset, oldest_mmap = self.lru_cache.popitem(last=False)
        oldest_mmap.close()

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise IndexError("Slicing with steps is not supported.")

            if key.start < 0:
                raise IndexError("Negative indexing is not supported.")

            if key.start >= key.stop:
                return b''

            start, stop = key.start, min(key.stop, self.file_size)
            start_page, stop_page = start // self.page_size, (stop - 1) // self.page_size + 1

            with self.lock:
                # Check if start and stop are within the same page
                if start_page == stop_page - 1:
                    self._ensure_mapped(start, stop - start)
                    offset = start // self.page_size * self.page_size
                    start_index = start - offset
                    stop_index = stop - offset
                    return self.lru_cache[offset][start_index:stop_index]

                # If the slice spans multiple pages, accumulate them
                self._ensure_mapped(start, stop - start)
                result = bytearray(stop - start)
                for page in range(start_page, stop_page):
                    self._ensure_mapped(page * self.page_size, 1, move_to_end=False)
                    offset = page * self.page_size
                    local_start = max(offset - start, 0)
                    start_index = max(start - offset, 0)
                    stop_index = min(stop, offset + self.page_size) - offset
                    result[local_start:local_start + stop_index - start_index] = self.lru_cache[offset][start_index:stop_index]
                return result
        elif isinstance(key, int):
            if key < 0:
                raise IndexError("Negative indexing is not supported.")

            if key >= self.file_size:
                raise IndexError("Index out of bounds.")

            with self.lock:
                self._ensure_mapped(key, 1)
                offset = key // self.page_size * self.page_size
                return self.lru_cache[offset][key - offset]

        raise TypeError(f"Index must be a slice or int (got {type(key)}).")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise IndexError("Slicing with steps is not supported.")

            if key.start < 0:
                raise IndexError("Negative indexing is not supported.")

            start, stop = key.start, min(key.stop, self.file_size)
            start_page, stop_page = start // self.page_size, (stop - 1) // self.page_size + 1
            length = stop - start

            if length != len(value):
                raise IndexError("Value length must match slice length.")

            if key.stop <= key.start:
                return

            with self.lock:
                self._ensure_mapped(start, length)
                value_offset = 0
                for page in range(start_page, stop_page):
                    self._ensure_mapped(page * self.page_size, 1, move_to_end=False)
                    page_offset = page * self.page_size
                    start_index = max(start - page_offset, 0)
                    chunk_size = min(stop - start, self.page_size - start_index)
                    self.lru_cache[page][start_index:start_index + chunk_size] = value[value_offset:value_offset + chunk_size]
                    start += chunk_size
                    value_offset += chunk_size
                return
        elif isinstance(key, int):
            if key < 0:
                raise IndexError("Negative indexing is not supported.")

            if key >= self.file_size:
                raise IndexError("Index out of bounds.")

            with self.lock:
                self._ensure_mapped(key, 1)
                page_offset = key // self.page_size * self.page_size
                mmap_obj = self.lru_cache[page_offset]
                mmap_obj[key - page_offset] = value
                return

        raise TypeError(f"Index must be a slice or int (got {type(key)}).")


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
