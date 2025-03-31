import numpy as np

class _WavFile():
    def __init__(self, filename):
        self.filename = filename
        self._get_metadata()

    def _get_metadata(self):
        with open(self.filename, 'rb') as file:
            header = file.read(12)
            if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
                raise ValueError("Invalid RIFF header or file format")
            
            self.offset = 12
            while True:
                subchunk_id = file.read(4)
                if not subchunk_id: 
                    raise ValueError("Unexpected end of file while searching for fmt subchunk")
                if subchunk_id == b'fmt ': break
                subchunk_size = int.from_bytes(file.read(4), 'little')
                file.seek(subchunk_size, 1)
            fmt_length = int.from_bytes(file.read(4), 'little')
            fmt_chunk = file.read(fmt_length)
            self.format = int.from_bytes(fmt_chunk[:2], 'little')
            if self.format != 1:
                raise NotImplementedError("Only PCM format supported")
            
            self.channels = int.from_bytes(fmt_chunk[2:4], 'little')
            self.sample_rate = int.from_bytes(fmt_chunk[4:8], 'little')
            self.byte_rate = int.from_bytes(fmt_chunk[8:12], 'little')
            self.block_align = int.from_bytes(fmt_chunk[12:14], 'little')
            self.bits_per_sample = int.from_bytes(fmt_chunk[14:16], 'little')

            assert (self.byte_rate) == (self.sample_rate * self.channels * self.bits_per_sample // 8)
            
            self.offset = file.tell()  # 정확한 오프셋 설정
            while True:
                subchunk_id = file.read(4)
                if not subchunk_id: 
                    raise ValueError("Unexpected end of file while searching for data subchunk")
                if subchunk_id == b'data': break
                subchunk_size = int.from_bytes(file.read(4), 'little')
                file.seek(subchunk_size, 1)    
            self.data_size = int.from_bytes(file.read(4), 'little') // self.block_align
            self.offset = file.tell()  # 데이터 시작 위치를 정확히 업데이트
        self.max  = 1
        self.mean = 0
        self.std  = 1
        _total = self.__getitem__(slice(0, self.data_size))
        self.max  = np.max(np.abs(_total))
        self.mean = np.mean(_total)
        self.std  = np.std(_total)
        # print(f"File: {self.filename[:8]}, Channels: {self.channels}, Sample Rate: {self.sample_rate}, Bit Depth: {self.bits_per_sample}, Data Size: {self.data_size}, Duration: {self.data_size / self.sample_rate:.2f} sec")
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            start = idx
            stop = idx + 1
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.data_size)
            if step != 1: 
                raise ValueError("Slice step not supported")
        else:
            raise ValueError("Invalid index type")
        
        with open(self.filename, 'rb') as file:
            file.seek(self.offset + start * self.block_align)
            data = file.read(self.block_align * (stop - start))
            if len(data) < (self.block_align * (stop - start)):
                raise IndexError("Index out of range")
        
        if self.bits_per_sample == 8: dtype = np.dtype(np.uint8) # 8비트 오디오는 일반적으로 부호 없는 데이터
        elif self.bits_per_sample == 16: dtype = np.dtype(np.int16)
        elif self.bits_per_sample == 32: dtype = np.dtype(np.int32)
        else: raise NotImplementedError("Unsupported bit depth")
        dtype = dtype.newbyteorder('<')  # 리틀 엔디안으로 바이트 순서 변경

        # 데이터 버퍼 읽기 및 정규화
        data = np.frombuffer(data, dtype=dtype).astype(np.float32)
        
        # 채널을 나누고 평균값 계산 (모노로 변환)
        data = data.reshape(-1, self.channels).transpose()[:1]
        # data = data.reshape(self.channels, -1).mean(axis=0, keepdims=True)

        # 비트 깊이에 따른 정규화
        if dtype == np.uint8: data = (data - 128) / 128.0  # 8비트 오디오는 0-255 범위이므로 중앙값을 기준으로 조정
        elif dtype == np.int16: data = data / (2 ** 15)  # 16비트 오디오는 -2^15 ~ 2^15-1 범위
        elif dtype == np.int32: data = data / (2 ** 31)  # 32비트 오디오는 -2^31 ~ 2^31-1 범위

        data = data * 0.95 / (self.max + 1e-8) # 최대값을 0.95로 정규화 (클리핑 방지)
        # data = (data - self.mean) / (self.std + 1e-8)  # 평균 0, 표준편차 1로 정규화
        return data

class _WavFileManager():
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
            cls._instance._files = {}
            cls._instance._ref_count = {}
        return cls._instance
    
    def _get_file(self, filename):
        _cls = type(self)
        if filename not in _cls._instance._files:
            self._instance._files[filename] = _WavFile(filename)
            _cls._instance._ref_count[filename] = 0
        _cls._instance._ref_count[filename] += 1
        return _cls._instance._files[filename]
    
    def _release_file(self, filename):
        _cls = type(self)
        if filename not in _cls._instance._files: return
        _cls._instance._ref_count[filename] -= 1
        if _cls._instance._ref_count[filename] <= 0:
            del _cls._instance._files[filename]
            del _cls._instance._ref_count[filename]

class WavFile():
    def __init__(self, filename):
        self.filename = filename
        self._manager = _WavFileManager()
        self._file = self._manager._instance._get_file(filename)

        # Load metadata
        self.channels = self._file.channels
        self.sample_rate = self._file.sample_rate
        self.bits_per_sample = self._file.bits_per_sample
        self.data_size = self._file.data_size
        self.max = self._file.max
        self.mean = self._file.mean
        self.std = self._file.std
    def __len__(self): return len(self._file)
    def __getitem__(self, idx): return self._file[idx]
    def __del__(self): self._manager._instance._release_file(self.filename)