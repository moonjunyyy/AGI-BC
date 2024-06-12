import numpy as np

class WavFile():
    def __init__(self, filename):
        self.filename = filename
        self.read_header()
        # print(self.format, self.channels, self.sample_rate, self.byte_rate, self.block_align, self.bits_per_sample, self.data_size)

    def read_header(self):
        with open(self.filename, 'rb') as file:
            # Parse RIFF header
            header = file.read(12)
            if header[:4] != b'RIFF' or header[8:12] != b'WAVE': raise ValueError("Invalid RIFF header")
            riff_length = int.from_bytes(header[4:8], 'little')
            # Parse Subchunks until fmt subchunk
            self.offset = 12
            while True:
                subchunk_id = file.read(4)
                if subchunk_id == b'fmt ': break
                subchunk_size = int.from_bytes(file.read(4), 'little')
                file.seek(subchunk_size, 1)
            # if file.read(4) != b'fmt ': raise ValueError("Invalid fmt subchunk")
            fmt_length = int.from_bytes(file.read(4), 'little')
            fmt_chunk = file.read(fmt_length)
            self.format = int.from_bytes(fmt_chunk[:2], 'little')
            if self.format != 1: raise NotImplementedError("Only PCM format supported")
            self.channels = int.from_bytes(fmt_chunk[2:4], 'little')
            self.sample_rate = int.from_bytes(fmt_chunk[4:8], 'little')
            self.byte_rate = int.from_bytes(fmt_chunk[8:12], 'little')
            self.block_align = int.from_bytes(fmt_chunk[12:14], 'little')
            self.bits_per_sample = int.from_bytes(fmt_chunk[14:16], 'little')
            # Parse Subchunks until data subchunk
            self.offset = self.offset + 8 + fmt_length
            while True:
                subchunk_id = file.read(4)
                if subchunk_id == b'data': break
                subchunk_size = int.from_bytes(file.read(4), 'little')
                file.seek(subchunk_size, 1)
            # Parse data subchunk
            self.data_size = int.from_bytes(file.read(4), 'little') // self.block_align
            self.offset = file.tell()

    def __len__(self):
        return self.data_size // self.block_align

    def __getitem__(self, idx) -> np.ndarray:
        if isinstance(idx, int):
            start = idx
            stop = idx + 1
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.data_size)
            if step != 1: raise ValueError("Slice step not supported")
        elif isinstance(idx, list) or isinstance(idx, tuple) or isinstance(idx, np.ndarray):
            raise NotImplementedError("Index list not supported")
        else: raise ValueError("Invalid index type")
        with open(self.filename, 'rb') as file:
            file.seek(self.offset + start * self.block_align, 0)
            data = file.read(self.block_align * (stop - start))
            if len(data) < (self.block_align * (stop - start)): raise IndexError("Index out of range")
        if self.bits_per_sample // 8 == 1: dtype = np.int8
        elif self.bits_per_sample // 8 == 2: dtype = np.int16
        elif self.bits_per_sample // 8 == 4: dtype = np.int32
        data = np.frombuffer(data, dtype=dtype).astype(dtype=np.float32)
        data = data.reshape(self.channels, -1).mean(axis=0, keepdims=True)
        data = data * 1 / (2 ** (self.bits_per_sample - 1))
        return data