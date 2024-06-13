from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p, CDLL
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

import numpy as np

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch import multiprocessing
from torch._dynamo.testing import rand_strided

from torch._inductor import ir
from torch._inductor.codecache import (
    CppCodeCache,
    CUDACodeCache,
    DLLWrapper,
    get_hash,
    PyCodeCache,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue

    from torch._inductor.select_algorithm import TritonTemplateCaller

from . import config
from .runtime.runtime_utils import do_bench_cpu, do_bench_gpu
from .virtualized import V

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
EXIT_HANDLER_REGISTERED = False

log = logging.getLogger(__name__)


# Used to synchronize between parent and child processes
class Ping:
    pass


class Pong:
    pass


class NonzeroWorkspaceNotSupportedError(Exception):
    pass


@contextlib.contextmanager
def set_cuda_visible_device(device: Optional[int]):
    """
    Context manager to set the CUDA_VISIBLE_DEVICES environment variable to the
    specified single device. If device is None, don't manipulate the environment.
    """
    if device is None:
        yield
        return

    current = os.environ.get(CUDA_VISIBLE_DEVICES)
    os.environ[CUDA_VISIBLE_DEVICES] = str(device)
    try:
        yield
    finally:
        if current is None:
            del os.environ[CUDA_VISIBLE_DEVICES]
        else:
            os.environ[CUDA_VISIBLE_DEVICES] = current


@dataclasses.dataclass
class TuningProcess:
    """
    Abstraction for launching a helper process to benchmark kernels. Spawns
    the parent process and uses multiprocessing queues to send benchmark
    requests and return results.
    """

    device: Optional[int] = None
    process: Optional[BaseProcess] = None
    request_queue: Optional[Queue[Any]] = None
    response_queue: Optional[Queue[Any]] = None

    @staticmethod
    def process_main(
        request_queue: Queue[Any],
        response_queue: Queue[Any],
    ) -> None:
        """
        Entry point for the child process.
        """
        log.debug(
            "Entering TuningProcess child. Visible devices = %s",
            os.environ.get(CUDA_VISIBLE_DEVICES),
        )
        try:
            TuningProcess.workloop(request_queue, response_queue)
        except Exception as ex:
            log.exception("Exception in TuningProcess")

    @staticmethod
    def workloop(request_queue: Queue[Any], response_queue: Queue[Any]) -> None:
        """
        Work loop for the benchmarking subprocess.
        """
        while True:
            obj = request_queue.get()

            if obj is None:
                break  # None is a sentinel for the child to terminate
            elif isinstance(obj, Ping):
                response_queue.put(Pong())
            elif isinstance(obj, BenchmarkRequest):
                response_queue.put(obj.benchmark())
            else:
                raise RuntimeError(f"Invalid request type {type(obj)}")

    def valid(self) -> bool:
        """
        True if the sub-process has been initialized.
        """
        return (
            self.process is not None
            and self.request_queue is not None
            and self.response_queue is not None
        )

    def clear(self) -> None:
        """
        Reset to an uninitialized state.
        """
        self.process = self.request_queue = self.response_queue = None

    def initialize(self) -> None:
        """
        Create child process, request/response queues, and do the warm up.
        Set the environment to make only the provided GPU device visible
        to the process.
        """
        if self.valid():
            return

        # cuda runtime does not work with "fork", use "spawn" to start processes.
        ctx = multiprocessing.get_context("spawn")
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()

        self.process = ctx.Process(
            target=self.process_main,
            args=(
                self.request_queue,
                self.response_queue,
            ),
        )
        assert self.process is not None
        with set_cuda_visible_device(self.device):
            self.process.start()

    def put(self, obj: Any) -> None:
        """
        Push a work item to the child process.
        """
        # In case of a prior crash, ensure the subprocess is running
        self.initialize()
        assert self.request_queue is not None
        self.request_queue.put(obj)

    def get(
        self, result_timeout=120.0, graceful_timeout=3.0, terminate_timeout=1.0
    ) -> Any:
        """
        Get a response from the child process. Raises queue.Empty on timeout
        or if the process dies.

        This method is (so far) only used by TuningProcessPool, where torch._inductor.config entries are being used
        to populate the timeouts:

        Arguments:

            @param result_timeout: Timeout in seconds, defaults to 120.0 or to
                                   config.max_autotune_subproc_result_timeout_seconds when called by TuningProcessPool
            @param graceful_timeout: Timeout in seconds to allow graceful shutdown (SIGTERM is sent after this time).
                                    Defaults to 3.0 or to config.max_autotune_subproc_graceful_timeout_seconds
            @param terminate_timeout: Timeout in seconds after SIGTERM, until we send SIGKILL if the process
                                      remains alive. Defaults to 1.0 or to
                                      config.max_autotune_subproc_terminate_timeout_seconds.
        Returns:
            A response from the child process (Any type)
        """
        assert self.process is not None
        assert self.response_queue is not None
        while True:
            try:
                remaining_timeout = result_timeout
                res = None
                while remaining_timeout is not None and remaining_timeout >= 1.0:
                    remaining_timeout -= 0.5
                    try:
                        res = self.response_queue.get(timeout=0.5)
                        break
                    except queue.Empty:
                        if not self.process.is_alive():
                            raise  # is being caught a few lines below
                if res is None:
                    res = self.response_queue.get(timeout=remaining_timeout)
                return res
            except queue.Empty:
                status = self.process.exitcode
                if status is None:
                    self.kill(
                        graceful_timeout=graceful_timeout,
                        terminate_timeout=terminate_timeout,
                    )
                else:
                    # child process crashed
                    self.clear()
                raise

    def terminate(self) -> None:
        """
        Signal the child process to terminate.
        """
        if self.valid():
            assert self.process is not None
            assert self.request_queue is not None
            self.request_queue.put(None)

    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
        if self.process is not None:
            self.process.join()
            self.clear()

    def kill(self, graceful_timeout=5.0, terminate_timeout=1.0) -> None:
        # Tries to kill the process, using a graceful_timeout in which the process
        # is allowed to exit gracefully. If the process is still alive,
        # it will be terminated. If that is not sufficient to end it
        # within terminate_timeout seconds, it will be killed.
        if self.process is not None:
            self.terminate()
            self.process.join(timeout=graceful_timeout)
            if self.process.is_alive():
                log.warning(
                    "Sending SIGTERM to process with PID %d",
                    self.process.pid,
                )
                self.process.terminate()
                self.process.join(timeout=terminate_timeout)
                if self.process.is_alive():
                    log.error(
                        "Sending SIGKILL to process with PID %d",
                        self.process.pid,
                    )
                    self.process.kill()  # This should definitely end the process
            self.clear()


@dataclasses.dataclass
class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """

    processes: Optional[queue.Queue[TuningProcess]] = None
    executor: Optional[ThreadPoolExecutor] = None

    def initialize(self) -> None:
        """
        Start the child processes.
        """
        assert (self.processes is None) == (self.executor is None)
        if self.processes is not None:
            return

        devices = self.get_device_list()
        log.debug("Sub-process autotune device list: %s", devices)

        # Launch the child processes and push a msg to "warm up"
        self.processes = queue.Queue()
        for device in devices:
            p = TuningProcess(device=device)
            p.initialize()
            p.put(Ping())
            self.processes.put(p)

        # Wait for the initialization to finish
        for p in self.processes.queue:
            assert isinstance(p.get(result_timeout=None), Pong)

        # Use a thread pool to manage distributing work to the subprocesses.
        # Threads block on an available process, so it makes sense to match
        # the number of threads with the number of devices.
        self.executor = ThreadPoolExecutor(max_workers=len(devices))

        # Register the exit handler for the parent process so it will terminate
        # the child processes.
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            atexit.register(self.terminate)

    def get_device_list(self) -> Sequence[Optional[int]]:
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            # Don't use multiple devices
            return [None]

        count = torch.cuda.device_count()

        # If the user specified the visible devices in the env, use those.
        if CUDA_VISIBLE_DEVICES in os.environ:
            devices = [int(d) for d in os.environ[CUDA_VISIBLE_DEVICES].split(",")]
            assert len(devices) <= count
            return devices

        return list(range(count))

    def terminate(self) -> None:
        """
        Signal all child processes to terminate.
        """
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

        if self.processes is not None:
            for p in self.processes.queue:
                p.terminate()
            for p in self.processes.queue:
                p.wait()
            self.processes = None

    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        assert choice.bmreq is not None
        assert self.processes is not None

        process = self.processes.get()
        process.put(choice.bmreq)
        try:
            return process.get(
                config.max_autotune_subproc_result_timeout_seconds,
                config.max_autotune_subproc_graceful_timeout_seconds,
                config.max_autotune_subproc_terminate_timeout_seconds,
            )
        except queue.Empty:
            warnings.warn(
                f"Failed to benchmark choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # set to INF so this choice will be ignored
            return float("inf")
        finally:
            self.processes.put(process)

    def benchmark(
        self,
        choices: List[TritonTemplateCaller],
    ) -> Dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """
        assert self.processes is not None, "Tuning process pool is not initialized"
        assert self.executor is not None

        results = {}

        # Use a ThreadExecutorPool to spread the work across the subprocesses and
        # to grab subprocesses as soon as they're free.
        for choice, result in zip(choices, self.executor.map(self.target, choices)):
            results[choice] = result

        return results


tuning_pool = TuningProcessPool()


LayoutOrBuffer = Union[ir.Layout, ir.Buffer]


@dataclasses.dataclass
class TensorMeta:
    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: Optional[str] = None

    @classmethod
    def from_irnodes(
        cls, irnodes: Union[LayoutOrBuffer, Sequence[LayoutOrBuffer]]
    ) -> Union[TensorMeta, List[TensorMeta]]:
        if isinstance(irnodes, Sequence):
            result: List[Any] = [cls.from_irnodes(x) for x in irnodes]
            assert all(isinstance(x, TensorMeta) for x in result)
            return result

        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer("fake", node)

        dtype = node.get_dtype()
        assert dtype is not None

        return TensorMeta(
            device=node.get_device(),
            dtype=dtype,
            sizes=V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            strides=V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            offset=V.graph.sizevars.size_hint(
                node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
            name=node.get_name(),
        )

    def to_tensor(self) -> torch.Tensor:
        return rand_strided(
            self.sizes,
            self.strides,
            device=self.device,
            dtype=self.dtype,
            extra_size=self.offset,
        )


@dataclasses.dataclass
class BenchmarkRequest:
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.

    Important: Instances of this class and subclasses have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
    ):
        # the kernel name defined in the module
        self.kernel_name = kernel_name

        if isinstance(input_tensor_meta, TensorMeta):
            input_tensor_meta = [input_tensor_meta]
        self.input_tensor_meta = input_tensor_meta

        if isinstance(output_tensor_meta, (tuple, list)):
            assert len(output_tensor_meta) == 1
            output_tensor_meta = output_tensor_meta[0]
        self.output_tensor_meta = output_tensor_meta

        self.extra_args = extra_args

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        raise NotImplementedError

    def cleanup_run_fn(self) -> None:
        pass

    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        raise NotImplementedError

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        debug = log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()

        # create args and out tensor
        if output_tensor is None:
            assert len(input_tensors) == 0
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            output_tensor = self.output_tensor_meta.to_tensor()

        if debug:
            create_tensor_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            start_ts = time.time()
        try:
            fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor)
        except NonzeroWorkspaceNotSupportedError:
            # Skipping all ops with nonzero workspace requirements
            log.info("Skipping op due to nonzero workspace requirement")
            return float("inf")

        if debug:
            load_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            start_ts = time.time()

        out = self.do_bench(fn, *input_tensors, output_tensor)

        if debug:
            bench_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            log.debug(
                "InChildProcess %s: load %f, create tensor %f, bench %f",
                str(self),
                load_elapse,  # type: ignore[possibly-undefined]
                create_tensor_elapse,  # type: ignore[possibly-undefined]
                bench_elapse,
            )
        self.cleanup_run_fn()
        return out


class TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file so that the TuningProcess
    sub-process knows how to unpickle these objects.
    """

    def __init__(self, value: Optional[float] = None) -> None:
        self.value = value

    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        if self.value is None:
            raise Exception("Failed to run")  # noqa: TRY002
        return self.value


class GPUDeviceBenchmarkRequest(BenchmarkRequest):
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        device_idx_set = {
            tensor.device.index
            for tensor in [*input_tensors, output_tensor]
            if isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.device.index is not None
        }
        assert len(device_idx_set) <= 1, f"Can not mix devices {device_idx_set}"
        if len(device_idx_set) == 1:
            device_idx = next(iter(device_idx_set))
        else:
            device_idx = torch.cuda.current_device()

        with torch.cuda.device(device_idx):
            out = do_bench_gpu(fn)
            torch.cuda.synchronize()  # shake out any CUDA errors

        return out


class TritonBenchmarkRequest(GPUDeviceBenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put CUDA Tensors in here!
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        module_path: str,  # the path of the module defining the triton kernel
        module_cache_key: str,
        grid: List[int],
        num_stages: int,
        num_warps: int,
        matrix_instr_nonkdim: int = 0,  # only used for hip to choose the shape of mfma instruction.
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.grid = grid
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.matrix_instr_nonkdim = matrix_instr_nonkdim

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        log.debug(
            "benchmark module key: %s, path: %s",
            self.module_cache_key,
            self.module_path,
        )

        run_method = getattr(mod, self.kernel_name).run
        extra_args = list(self.extra_args)

        # Newer version of triton add warmup argument to JITFunction.run.
        # This code handles backward-compatibility.
        warmup_arg = {}
        import inspect

        if "warmup" in inspect.signature(run_method).parameters:
            warmup_arg["warmup"] = False

        from torch._C import _cuda_getCurrentRawStream as get_raw_stream

        if torch.version.hip and self.matrix_instr_nonkdim != 0:
            return functools.partial(
                run_method,
                *input_tensors,
                output_tensor,
                *self.extra_args,
                grid=self.grid,
                **warmup_arg,
                stream=get_raw_stream(self.output_tensor_meta.device.index),
            )
        else:
            return functools.partial(
                run_method,
                *input_tensors,
                output_tensor,
                *self.extra_args,
                grid=self.grid,
                **warmup_arg,
                stream=get_raw_stream(self.output_tensor_meta.device.index),
            )

    def precompile(self):
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        getattr(mod, self.kernel_name).precompile()

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.module_path=}, {self.module_cache_key=}"


class GroupedTritonBenchmarkRequest:
    def __init__(self, choices):
        super().__init__()
        self.choices = choices

    def do_benchmark(
        self,
        choice_to_callable: Dict[TritonTemplateCaller, Callable[[], Any]],
        target: float,
    ) -> Dict[TritonTemplateCaller, float]:
        timings: Dict[TritonTemplateCaller, float] = {}

        to_estimate = []
        for choice, callable in choice_to_callable.items():
            # backout of invalid choices. this covers choices
            # that crash during ptx generation, or choices that
            # are not supported on the current device.
            if not choice.valid:
                timings[choice] = float("inf")
                continue

            # initialize the choices before benchmarking, and
            # backout of choices that crash during compile time
            try:
                callable()
            except Exception:
                timings[choice] = float("inf")
                continue
            
            to_estimate.append(choice)
        
        @functools.lru_cache(None)
        def get_cache_size():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            return properties.l2CacheSize

        cache = torch.empty(int(get_cache_size() // 4), dtype=torch.int, device="cuda")

        def interleaved_timing(choices, iters):
            choice_event_pairs = {
                choice: [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(iters)
                ]
                for choice in choices
            }

            for iter in range(iters):
                for choice, event_pairs in choice_event_pairs.items():
                    start_event, end_event = event_pairs[iter]
                    callable = choice_to_callable[choice]
                    cache.zero_()
                    start_event.record()
                    callable()
                    end_event.record()
            torch.cuda.synchronize()

            return {
                choice: min(
                    [
                        start_event.elapsed_time(end_event)
                        for start_event, end_event in event_pairs
                    ]
                )
                for choice, event_pairs in choice_event_pairs.items()
            }

        estimation_iters = 5
        estimates = interleaved_timing(to_estimate, estimation_iters)
        timings.update(estimates)

        to_benchmark = []
        target = min(target, estimates[min(estimates, key=estimates.__getitem__)])
        for choice, estimate in estimates.items():
            if estimate * 0.975 <= target:
                to_benchmark.append(choice)
        
        benchmarking_iters = 15
        benchmarks = interleaved_timing(to_benchmark, benchmarking_iters)
        for choice, benchmark in benchmarks.items():
            timings[choice] = min(timings[choice], benchmark)

        del cache

        return timings


    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
        target: float = float("inf"),
    ) -> Dict[TritonTemplateCaller, float]:
        timings: Dict[TritonTemplateCaller, float] = {}

        if self.choices == []:
            return timings

        # generate inputs/output tensors using choices[0] as
        # the default. every choice should take the same exact
        # inputs/outputs
        if output_tensor is None:
            assert len(input_tensors) == 0
            example_bmreq = self.choices[0].bmreq
            input_tensors = tuple(
                x.to_tensor() for x in example_bmreq.input_tensor_meta
            )
            output_tensor = example_bmreq.output_tensor_meta.to_tensor()

        choice_to_callable = {}
        for choice in self.choices:
            _callable = choice.bmreq.make_run_fn(
                *input_tensors, output_tensor=output_tensor
            )
            choice_to_callable[choice] = _callable

        timings.update(self.do_benchmark(choice_to_callable, target))
        return timings


class CUDABenchmarkRequest(GPUDeviceBenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put CUDA Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.DLL: Optional[DLLWrapper] = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, "so")

    def precompile(self):
        # Prepopulate CUDACodeCache
        # may happen in separate Threadpool
        log.debug("Precompiling %s", self)
        CUDACodeCache.compile(self.source_code, "so")
        log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [
            c_void_p(tensor.data_ptr())
            for tensor in list(input_tensors) + [output_tensor]
        ]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=output_tensor.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

    def update_workspace_size(self) -> None:
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len({meta.name for meta in self.input_tensor_meta})
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        run_method(
            *args,  # input ptrs and output ptrs
            *self.extra_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        torch.cuda.synchronize()  # shake out any CUDA errors
        self.workspace_size = c_workspace_size.value
        log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",  # noqa: B950
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = CUDACodeCache.load(
                self.source_code, "so"
            )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
        self.workspace = None

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"


class CPUDeviceBenchmarkRequest(BenchmarkRequest):
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        return do_bench_cpu(fn)


class CppBenchmarkRequest(CPUDeviceBenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.hash_key = get_hash(source_code)
        self.DLL: Optional[Union[CDLL, ModuleType]] = None

    def precompile(self):
        # Prepopulate CppCodeCache
        # may happen in separate Threadpool
        log.debug("Precompiling %s", self)
        CppCodeCache.load(self.source_code, cuda=False)
        log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        # TODO(jgong5): use CppPythonBindingsCodeCache for better binding perf
        self.DLL = CppCodeCache.load(self.source_code, cuda=False)
        args = [tensor.data_ptr() for tensor in list(input_tensors) + [output_tensor]]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        run_method.argtypes = [ctypes.c_ulonglong] * len(args)

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
        )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()

    def __str__(self) -> str:
        return f"{self.kernel_name=}"


def benchmark_in_sub_process(
    choices: List[TritonTemplateCaller],
) -> Dict[TritonTemplateCaller, float]:
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
    return tuning_pool.benchmark(choices)
