import gc
import torch
import time
import pickle
import random
import asyncio
# import secrets
import threading
import socket
from torch import Tensor
import multiprocessing as mp
from typing import Callable, Tuple, Any, Sequence, List, Dict
from utils.log import Log

random.seed(int(time.time() * 1e6))

CHUNK_SIZE = 2**16 # 4KB
SUCCESS = 0x0000
FAILURE = 0x0001
ACK = 0xFFFF
MAX_RETRY = 1000
SLEEP_TIME = 1e-6 # 1 us
SOCKET_LIFETIME = 10 # 20 seconds
TIMEOUT = 5 # 1 second

def _thread_and_wait(target: Callable[[], Any], args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
    thread = threading.Thread(target=target, args=args, kwargs=kwargs)
    thread.start(); thread.join();

def _sleep_time():
    # return SLEEP_TIME + SLEEP_TIME * secrets.randbelow(100)/1000 * ((secrets.randbelow(1) - 0.5) * -2) # Random sleep time between +- 10% of SLEEP_TIME
    return SLEEP_TIME + SLEEP_TIME * random.random() * ((random.random() - 0.5) * -2) # Random sleep time between +- 10% of SLEEP_TIME

class _Socket:
    is_socket_closed = lambda sock: sock.fileno() == -1

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
        
    @staticmethod
    async def _recv(sock: socket.socket, length: int) -> Tuple[int, bytes]:
        recv_size = 0; data = [];
        retry = 0
        while recv_size < length:
            if retry > MAX_RETRY: return FAILURE, b''
            try:
                if _Socket.is_socket_closed(sock): raise RuntimeError("Socket is closed.")
                chunk = sock.recv(min(CHUNK_SIZE, length - recv_size))
                if not chunk: return FAILURE, b'' #continue;
            except Exception as e:
                return FAILURE, b''
            recv_size += len(chunk)
            data.append(chunk)
        return SUCCESS, b''.join(data)

    @staticmethod
    async def _send(sock: socket.socket, data: bytes) -> int:
        sent_size = 0
        while sent_size < len(data):
            try:
                if _Socket.is_socket_closed(sock): raise RuntimeError("Socket is closed.")
                sent_size += sock.send(data[sent_size:min(sent_size+CHUNK_SIZE, len(data))])
            except Exception as e:
                return FAILURE
        return SUCCESS

    @staticmethod
    async def send_data(sock: socket.socket, data: Any, timeout=TIMEOUT) -> int:
        o_timeout = sock.gettimeout()
        sock.settimeout(timeout)
        dump = pickle.dumps(data)
        length = len(dump).to_bytes(4, 'big')
        status = await _Socket._send(sock, length)
        if status == FAILURE: return FAILURE;
        # status, ack = _Socket._recv(sock, 4)
        # if status == FAILURE: return FAILURE;
        # if int.from_bytes(ack, 'big') != ACK: return FAILURE;
        status = await _Socket._send(sock, dump)
        if status == FAILURE: return FAILURE;
        # status, ack = _Socket._recv(sock, 4)
        # if status == FAILURE: return FAILURE;
        # if int.from_bytes(ack, 'big') != ACK: return FAILURE;
        sock.settimeout(o_timeout)
        return SUCCESS

    @staticmethod
    async def recv_data(sock: socket.socket, timeout=TIMEOUT) -> Tuple[int, Any]:
        o_timeout = sock.gettimeout()
        sock.settimeout(timeout)
        status, length = await _Socket._recv(sock, 4)
        if status == FAILURE: return FAILURE, None;
        # status = _Socket._send(sock, ACK.to_bytes(4, 'big'))
        # if status == FAILURE: return FAILURE, None;
        length = int.from_bytes(length, 'big')
        status, data = await _Socket._recv(sock, length)
        if status == FAILURE: return FAILURE, None;
        # status = _Socket._send(sock, ACK.to_bytes(4, 'big'))
        # if status == FAILURE: return FAILURE, None;
        sock.settimeout(o_timeout)
        return SUCCESS, pickle.loads(data)
    
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def close(self):
        self.sock.close()

class _SocketServer(_Socket):
    def __init__(self, port: int):
        self.port = port
        self.conn = None
        self.event_callback = {}
        self.sock = self.open_socket()

    def open_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host = socket.gethostname()
        sock.bind((host, self.port))
        sock.settimeout(TIMEOUT)
        sock.listen()
        return sock
    
    def destroy_socket(self, sock: socket.socket):
        sock.shutdown(socket.SHUT_RDWR)
        sock.close(); del sock;

    def close(self):
        self.sock.close()

    def add_event_callback(self, event: Any, callback: Callable[[Any], Any]):
        self.event_callback[event] = callback

    def _event_loop(self):
        async def _loop():
            while True:
                try:
                    self.conn, self.addr = self.sock.accept()
                    self.conn.settimeout(TIMEOUT)
                except Exception as e: continue;
                try:
                    status, req = await self.recv_data(self.conn)
                    if status == FAILURE: raise RuntimeError("Failed to receive data.");
                    if req in self.event_callback:
                        callback = self.event_callback[req]; await callback(self.conn);
                except Exception as e: continue;
                finally: 
                    self.conn.close(); self.conn = None;
                while True:
                    try:
                        self.destroy_socket(self.sock)
                        self.sock = self.open_socket()
                        break;
                    except Exception as e: continue;
        asyncio.run(_loop())

class _SocketClient(_Socket):
    def __init__(self, port: int):
        super().__init__(port)
        self.sock.settimeout(TIMEOUT)
        host = socket.gethostname()
        try: self.sock.connect((host, self.port))
        except Exception as e: raise RuntimeError("Failed to connect to server");
    
    def close(self):
        self.sock.close()

class _PrefetchWorker(mp.Process):
    def __init__(self, port: int, name: str = None):
        super().__init__(daemon=True)
        self.port = port
        self.sock = _SocketServer(self.port)

        self.dataset = None
        self.transform = None
        self.indices = []
        self.output_buffer = []
        self.n_iter = 1

        self.data_lock = None   # Lock for dataset, transform, and indices, Lock cannot be pickled, so it must be initialized in the run method.
        self.output_lock = None # Lock for output_buffer, Lock cannot be pickled, so it must be initialized in the run method.

    async def _flush(self, sock:socket.socket):
        with self.data_lock:
            self.dataset = None
            self.transform = None
            self.indices = []
            self.n_iter = 1
            with self.output_lock:
                self.output_buffer = []
        await _Socket.send_data(sock, SUCCESS)

    def _worker_thread(self):
        '''
        Worker thread to prefetch data. Runs forever, until the process is terminated.
        When the dataset is None, the worker thread will wait until the dataset is set.
        '''
        while True:
            time.sleep(_sleep_time());
            try:
                with self.data_lock:
                    if self.dataset is None: continue;
                    if len(self.indices) == 0:
                        with self.output_lock:
                            self.output_buffer.append(None);
                        self.dataset = None
                        continue;
                    else: indice = self.indices.pop(0);
            except Exception as e:
                time.sleep(_sleep_time()); continue;
            while True:
                try: batch = asyncio.run(self._build_batch(indice)); break;
                except Exception as e: continue;
            while True:
                time.sleep(_sleep_time());
                with self.output_lock:
                    if len(self.output_buffer) > 0: continue;
                    self.output_buffer.append(batch); break;
                
    async def _get_sample(self, idx:int) -> Tuple[Tensor,] | List[Tensor,] | Dict[Any, Tensor]:
        ret = self.dataset[idx]; return ret, type(ret);
    
    async def _build_batch(self, indice: Sequence[int]) -> Tuple[Tensor,] | List[Tensor,] | Dict[Any, Tensor]:
        batch = []; typename = None;
        _iterated_indices = indice * self.n_iter
        # batch = [self._get_sample(index) for index in _iterated_indices]
        batch = await asyncio.gather(*[self._get_sample(index) for index in _iterated_indices])
        typename = batch[0][1]; batch = tuple(zip(*batch))[0];
        if typename is None: raise RuntimeError(f"{self.name} build_batch : Error occured building batch : type is {typename}")
        elif typename is list or typename is tuple:
            try:
                batch = list(zip(*batch))
                batch = [torch.stack(b) if isinstance(b[0], Tensor) else torch.tensor(b) for b in batch]
            except Exception as e:
                print(f"{self.name} build_batch : Error occured building batch\n{e}")
        elif typename is dict:
            try:
                keys = list(batch[0].keys())
                batch = {k: torch.stack([s[k] for s in batch]) if isinstance(batch[0][k], Tensor) else torch.tensor([s[k] for s in batch]) for k in keys}
            except Exception as e:
                print(f"{self.name} build_batch : Error occured building batch\n{e}")
        else: batch = torch.stack(batch)
        if self.transform is not None:
            try: batch = self.transform(batch)
            except Exception as e:
                print(f"{self.name} build_batch : Error occured building batch\n{e}")
        return batch

    def run(self) -> None:
        self.data_lock = threading.Lock()   # Lock for dataset, transform, and indices. Lock cannot be pickled, so it must be initialized in the run method.
        self.output_lock = threading.Lock() # Lock for output_buffer. Lock cannot be pickled, so it must be initialized in the run method.
        self.sock.add_event_callback("set_metadata", self._set_metadata)
        self.sock.add_event_callback("get_batch", self._get_batch)
        self.sock.add_event_callback("flush", self._flush)
        self.sock.add_event_callback("ping", self._ping)
        _worker=None; _server=None;
        # while True:
        #     if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=self._worker_thread, name=f'_worker_{self.port}', daemon=True)
        _worker.start();
            # if _server is None or not _server.is_alive():
        _server = threading.Thread(target=self.sock._event_loop, name=f'_server_{self.port}', daemon=True)
        _server.start();
            # time.sleep(sleep_time())
        _worker.join(); _server.join();

    async def _ping(self, conn: socket.socket):
        try:
            status = await _Socket.send_data(conn, SUCCESS)
            if status == FAILURE: raise RuntimeError("Failed to ping.");
        except Exception as e: await _Socket.send_data(conn, FAILURE); raise RuntimeError("Failed to ping.");

    async def _set_metadata(self, conn: socket.socket):
        try:
            result, metadata = await _Socket.recv_data(conn)
            if result == FAILURE: raise RuntimeError("Failed to set metadata.");
            with self.data_lock:
                self.dataset, self.transform, self.indices, self.n_iter = metadata
        except Exception as e: raise RuntimeError("Failed to set metadata.");

    async def _get_batch(self, conn: socket.socket):
        try:
            with self.output_lock:
                if len(self.output_buffer) > 0: send_data = self.output_buffer[0];
                else: send_data = FAILURE;
        except Exception as e: send_data = FAILURE
        try:
            status = await _Socket.send_data(conn, send_data);
            if status == FAILURE: raise RuntimeError("Failed to send data.");
            if send_data == FAILURE: return;
            with self.output_lock: self.output_buffer.pop(0);
        except Exception as e:
            status = await _Socket.send_data(conn, FAILURE);
            raise RuntimeError("Failed to send data.");
        
class Prefetcher:
    def __init__(self, n_workers: int, name: str = 'Prefetcher'):
        self.n_workers = n_workers
        self.n_iter = 1
        self.name = name
        self.ports = []
        self.workers = []
        while True:
            try:
                port = random.randint(10000, 2**16-1)
                # if _Socket.is_port_in_use(port): continue;
                worker = _PrefetchWorker(port)
                worker.start()
                self.ports.append(port) 
                self.workers.append(worker)
            except Exception as e: time.sleep(_sleep_time()); continue;
            if len(self.ports) == n_workers: break;
        self.batchsize = 0
        self._worker_cursor = 0

        self.dots = 3
        self.output_buffer = []
        self.output_lock = threading.Lock() # Prevent datarace when accessing the output buffer.
        self.flush_event = threading.Event()
        thread = threading.Thread(target=self.ping)
        thread.start(); thread.join();

    def check_workers(self) -> List[bool]:
        return [w.is_alive() if w.is_alive() else w.exitcode for w in self.workers]

    def start(self):
        [w.start() for w in self.workers]

    def print_waiting_dots(self):
        with self.output_lock:
            print(f"Wating for the workers to be ready{'.'*self.dots}    ", end='\r')
            self.dots = (self.dots + 1) % 4

    async def _ping(self, port: int):
        while True:
            try:
                sock = _SocketClient(port)
            except Exception as e: time.sleep(_sleep_time()); continue;
            try:
                status = await _Socket.send_data(sock.sock, "ping");
                if status == FAILURE: time.sleep(_sleep_time()); continue;
                status, res = await _Socket.recv_data(sock.sock)
                if res != SUCCESS | status == FAILURE : time.sleep(_sleep_time()); continue;
                break;
            except Exception as e:
                self.print_waiting_dots()
                time.sleep(_sleep_time()); continue;
            finally: sock.close();
    
    async def _ping_gather(self):
        await asyncio.gather(*[self._ping(worker.port) for worker in self.workers])
    
    def ping(self):
        asyncio.run(self._ping_gather())

    async def _send_metadata(self, command:str, metadata: Any, port: int):
        while True:
            try:
                sock = _SocketClient(port)
            except Exception as e: time.sleep(_sleep_time()); continue;
            try:
                status = await _Socket.send_data(sock.sock, command);
                if status == FAILURE: time.sleep(_sleep_time()); continue;
                status = await _Socket.send_data(sock.sock, metadata);
                if status == FAILURE: time.sleep(_sleep_time()); continue;
                break;
            except Exception as e: time.sleep(_sleep_time()); continue;
            finally: sock.close();

    async def _metadata_gather(self):
        await asyncio.gather(*[self._send_metadata("set_metadata", (self.dataset, self.transform, self.indices[n], self.n_iter), worker.port) for n, worker in enumerate(self.workers)])

    def send_metadata(self, command:str, metadata: Any):
        asyncio.run(self._metadata_gather())

    async def get_batch(self, worker_id) -> Tuple[Tensor,] | List[Tensor,] | Dict[Any,Tensor]:
        while True:
            try:
                sock = _SocketClient(self.workers[worker_id].port)
            except Exception as e: time.sleep(_sleep_time()); continue;
            try:
                status = await _Socket.send_data(sock.sock, "get_batch");
                if status == FAILURE: time.sleep(_sleep_time()); continue;
                status, data = await _Socket.recv_data(sock.sock)
                if status == FAILURE or data == FAILURE:time.sleep(_sleep_time()); continue;
                else: break;
            except Exception as e: time.sleep(_sleep_time()); continue;
            finally: sock.close();
        return data

    async def _flush(self, port: int):
        while True:
            try:
                sock = _SocketClient(port)
            except Exception as e:
                time.sleep(_sleep_time()); continue;
            try:
                status = await _Socket.send_data(sock.sock, "flush");
                if status == FAILURE: time.sleep(_sleep_time()); continue;
                status, res = await _Socket.recv_data(sock.sock)
                if res != SUCCESS | status == FAILURE : time.sleep(_sleep_time()); continue;
                break;
            except Exception as e: time.sleep(_sleep_time()); continue;
            finally: sock.close();
    
    async def _flush_gather(self):
        await asyncio.gather(*[self._flush(worker.port) for worker in self.workers])
    
    def flush(self):
        self.flush_event.set()
        asyncio.run(self._flush_gather())
        with self.output_lock:
            self.output_buffer = []
        self._worker_cursor = 0
        self.flush_event.clear()

    def load(self, dataset:Sequence[Any],
            transform:Callable[[Any,],Tuple[Tensor,...]],
            batchsize:int,
            n_iter:int=1,
            random:bool=False,
            sampler:torch.utils.data.Sampler=None,
            generator:torch.Generator=None) -> None:
        _thread_and_wait(target=self.flush)
        if sampler is None:
            if generator is None: self.generator = torch.Generator()
            else : generator = generator
            if random: sampler = torch.utils.data.RandomSampler(dataset, generator=self.generator)
            else: sampler = torch.utils.data.SequentialSampler(dataset)
        else: assert not random, 'Cannot use a custom sampler and random = True'
        self.batchsize = batchsize
        self.dataset = dataset
        self.transform = transform
        self.n_iter = n_iter
        self.sampler = sampler
    
    def _worker_num_generator(self):
        ret = self._worker_cursor
        self._worker_cursor = (self._worker_cursor + 1) % self.n_workers
        return ret

    def _fetch(self):
        '''
        Fetch data from the workers.
        Run this function until all workers sent a None.
        '''
        none_count = 0
        worker_id = self._worker_num_generator()
        while none_count < self.n_workers:
            if self.flush_event.is_set(): break;
            try: data = asyncio.run(self.get_batch(worker_id))
            except Exception as e:
                print(f"Error occured in _fetch : {e}");
                time.sleep(_sleep_time()); continue;
            worker_id = self._worker_num_generator();
            if data is None: none_count += 1; continue;
            while True:
                time.sleep(_sleep_time()); 
                with self.output_lock:
                    if len(self.output_buffer) > 1 : continue;
                    self.output_buffer.append(data); break;
        with self.output_lock: self.output_buffer.append(None)

    def __iter__(self):
        self.indices = list(self.sampler)
        self.indices = [[self.indices[i:i+self.batchsize] for i in range(n * self.batchsize, len(self.indices), self.n_workers * self.batchsize)] for n in range(self.n_workers)]
        _thread_and_wait(target=self.send_metadata, args=("set_metadata", (self.dataset, self.transform, self.indices[0], self.n_iter)))
        self.fetch_thread = threading.Thread(target=self._fetch)
        self.fetch_thread.start()
        return _PrefetcherIterator(self)

    def __del__(self):
        for worker in self.workers:
            worker.terminate()
            worker.join()

    def __len__(self):
        return len(self.sampler) // self.batchsize + (len(self.sampler) % self.batchsize != 0)
    
class _PrefetcherIterator:
    def __init__(self, prefetcher: Prefetcher):
        self.prefetcher = prefetcher

    def __next__(self) -> Tuple[Tensor,] | List[Tensor,] | Dict[Any,Tensor]:
        while True:
            time.sleep(_sleep_time())
            with self.prefetcher.output_lock:
                if len(self.prefetcher.output_buffer) > 0:
                    data = self.prefetcher.output_buffer.pop(0)
                    if data is None : raise StopIteration
                    return data
                else: continue;

    def __del__(self):
        _thread_and_wait(target=self.prefetcher.flush)
        torch.cuda.empty_cache()

    def __iter__(self):
        return self