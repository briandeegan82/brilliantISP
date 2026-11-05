"""Runtime-compiled CUDA/HIP device kernels via ctypes NVRTC/HIPRTC.

Compiler and driver shared libraries are loaded lazily; torch owns the
device context, memory, and stream — kernels launch on
``torch.cuda.current_stream(dev).cuda_stream`` after input tensors are
``record_stream``-protected. Compiled modules live in a thread-safe
bounded cache keyed by vendor/device/arch/source hash; eviction unloads
the driver module after device synchronization. Absent libraries raise
GpuUnavailableError; compile/launch failures raise GpuError with logs.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import sysconfig
import threading
from collections import OrderedDict
from pathlib import Path

from ..types import GpuError, GpuUnavailableError, require_torch
from ..device import backend_vendor

_KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels"
_CACHE_LIMIT = 16
_module_cache: OrderedDict = OrderedDict()
_cache_lock = threading.RLock()
_libs: dict = {}
_libs_lock = threading.Lock()


def load_source(name: str) -> bytes:
    return (_KERNEL_DIR / name).read_bytes()


def _site_packages() -> list[Path]:
    paths = []
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            paths.append(Path(p))
    return paths


def _load_lib(candidates: tuple[str, ...], package_globs: tuple[str, ...], what: str):
    with _libs_lock:
        if what in _libs:
            lib = _libs[what]
            if lib is None:
                raise GpuUnavailableError(f"{what} shared library is not available")
            return lib
        lib = None
        for sp in _site_packages():
            for pat in package_globs:
                for found in sorted(sp.glob(pat)):
                    try:
                        lib = ctypes.CDLL(str(found))
                        break
                    except OSError:
                        continue
                if lib is not None:
                    break
            if lib is not None:
                break
        if lib is None:
            for name in candidates:
                path = ctypes.util.find_library(name)
                if path:
                    try:
                        lib = ctypes.CDLL(path)
                        break
                    except OSError:
                        continue
        _libs[what] = lib
    if lib is None:
        raise GpuUnavailableError(
            f"{what} shared library is not available "
            f"(looked in packaged paths {package_globs} and system linker)"
        )
    return lib


def _cuda_compiler():
    return _load_lib(
        ("nvrtc",),
        ("nvidia/cuda_nvrtc/lib/libnvrtc.so*", "nvidia/cuda_nvrtc/lib64/libnvrtc.so*"),
        "NVRTC",
    )


def _cuda_driver():
    return _load_lib(("cuda",), (), "CUDA driver (libcuda)")


def _hip_compiler():
    return _load_lib(("hiprtc",), (), "HIPRTC")


def _hip_driver():
    return _load_lib(("amdhip64",), (), "AMD HIP driver (libamdhip64)")


def _set_prototypes(lib, specs: dict, what: str) -> None:
    for name, (restype, argtypes) in specs.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as exc:
            raise GpuUnavailableError(f"{what} missing required symbol {name}") from exc
        try:
            fn.restype = restype
            fn.argtypes = argtypes
        except AttributeError:
            continue


_NVRTC_SPECS = {
    "nvrtcCreateProgram": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]),
    "nvrtcCompileProgram": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]),
    "nvrtcGetProgramLogSize": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]),
    "nvrtcGetProgramLog": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
    "nvrtcGetPTXSize": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]),
    "nvrtcGetPTX": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
    "nvrtcDestroyProgram": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
}

_CU_SPECS = {
    "cuModuleLoadData": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]),
    "cuModuleGetFunction": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]),
    "cuModuleUnload": (ctypes.c_int, [ctypes.c_void_p]),
    "cuLaunchKernel": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]),
}

_HIPRTC_SPECS = {
    "hiprtcCreateProgram": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]),
    "hiprtcCompileProgram": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]),
    "hiprtcGetProgramLogSize": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]),
    "hiprtcGetProgramLog": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
    "hiprtcGetCodeSize": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]),
    "hiprtcGetCode": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
    "hiprtcDestroyProgram": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
}

_HIP_SPECS = {
    "hipModuleLoadData": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]),
    "hipModuleGetFunction": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]),
    "hipModuleUnload": (ctypes.c_int, [ctypes.c_void_p]),
    "hipModuleLaunchKernel": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]),
}

_proto_done: set = set()


def _proto(lib, specs, key, what):
    if key not in _proto_done:
        _set_prototypes(lib, specs, what)
        _proto_done.add(key)


def _bytes_array(strings):
    arr = (ctypes.c_char_p * len(strings))()
    for i, s in enumerate(strings):
        arr[i] = s
    return arr


def _check(rc, name):
    if rc != 0:
        raise GpuError(f"{name} failed: {rc}")


def _compile_program(compiler, prefix, source, options, code_kind):
    prog = ctypes.c_void_p()
    call = lambda name, *args: getattr(compiler, prefix + name)(*args)
    _check(call("CreateProgram", ctypes.byref(prog), source, b"reference_sort.cu",
                0, None, None), prefix + "CreateProgram")
    primary = None
    try:
        rc = call("CompileProgram", prog, len(options), _bytes_array(options))
        log = b""
        try:
            log_size = ctypes.c_size_t()
            _check(call("GetProgramLogSize", prog, ctypes.byref(log_size)),
                   prefix + "GetProgramLogSize")
            if log_size.value:
                buf = ctypes.create_string_buffer(int(log_size.value))
                _check(call("GetProgramLog", prog, buf), prefix + "GetProgramLog")
                log = buf.value
        except GpuError as exc:
            if rc != 0:
                raise GpuError(f"{prefix} compile failed (rc={rc}); {exc}") from exc
            raise
        if rc != 0:
            raise GpuError(
                f"{prefix} compile failed (rc={rc}):\n"
                f"{log.decode('utf-8', 'replace')}"
            )
        size = ctypes.c_size_t()
        _check(call(f"Get{code_kind}Size", prog, ctypes.byref(size)),
               prefix + f"Get{code_kind}Size")
        if not size.value:
            raise GpuError(f"{prefix} returned an empty {code_kind} image")
        image = ctypes.create_string_buffer(int(size.value))
        _check(call(f"Get{code_kind}", prog, image), prefix + f"Get{code_kind}")
        return bytes(image.raw)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        rc = call("DestroyProgram", ctypes.byref(prog))
        if rc != 0:
            error = GpuError(f"{prefix}DestroyProgram failed: {rc}")
            if primary is None:
                raise error
            primary.add_note(str(error))


def _nvrtc_compile(source: bytes, arch: str):
    nvrtc = _cuda_compiler()
    _proto(nvrtc, _NVRTC_SPECS, ("nvrtc", id(nvrtc)), "NVRTC")
    return _compile_program(
        nvrtc, "nvrtc", source,
        [f"--gpu-architecture=compute_{arch}".encode(), b"--fmad=false"], "PTX",
    )


def _hiprtc_compile(source: bytes, arch: str):
    hiprtc = _hip_compiler()
    _proto(hiprtc, _HIPRTC_SPECS, ("hiprtc", id(hiprtc)), "HIPRTC")
    return _compile_program(
        hiprtc, "hiprtc", source,
        [f"--offload-arch={arch}".encode(), b"-ffp-contract=off"], "Code",
    )


def _arch_for(device) -> str:
    torch = require_torch()
    props = torch.cuda.get_device_properties(device)
    if backend_vendor() == "amd":
        gcn = getattr(props, "gcnArchName", None)
        if not isinstance(gcn, str) or not gcn.startswith("gfx"):
            raise GpuUnavailableError(
                f"HIP device properties lack a usable gcnArchName ({gcn!r}); "
                "cannot target HIPRTC"
            )
        return gcn
    return f"{props.major}{props.minor}"


class _ModuleEntry:
    __slots__ = ("vendor", "device_index", "device", "handle", "funcs",
                 "driver", "lock", "lease_count", "retired")

    def __init__(self, vendor, device, handle, funcs, driver):
        self.vendor = vendor
        self.device = device
        self.device_index = device.index
        self.handle = handle
        self.funcs = funcs
        self.driver = driver
        self.lock = threading.Lock()
        self.lease_count = 0
        self.retired = False

    def unload(self) -> None:
        handle = self.handle
        if handle is None:
            return
        if self.vendor == "amd":
            rc = self.driver.hipModuleUnload(handle)
        else:
            rc = self.driver.cuModuleUnload(handle)
        if rc != 0:
            raise GpuError(f"module unload failed: {rc}")
        self.handle = None


class KernelHandle:
    """Raw function pointer plus a strong reference to its open lease."""

    __slots__ = ("lease", "name", "fn")

    def __init__(self, lease, name, fn):
        self.lease = lease
        self.name = name
        self.fn = fn


class _ModuleLease:
    """Context-managed lease on a cached module entry.

    ``funcs(name)`` returns a KernelHandle usable for launches while the
    lease is open; after the context exits every retained handle raises
    GpuError before any driver call.
    """

    __slots__ = ("_entry", "_open")

    def __init__(self, entry):
        self._entry = entry
        self._open = True

    def __enter__(self):
        if not self._open:
            raise GpuError("module lease is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            _release(self)
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"module lease cleanup failed: {cleanup}")
        return False

    def __call__(self, name: str) -> KernelHandle:
        return self.funcs(name)

    def funcs(self, name: str) -> KernelHandle:
        if not self._open:
            raise GpuError("module lease is closed")
        try:
            fn = self._entry.funcs[name]
        except KeyError:
            raise GpuError(f"module has no function {name!r}") from None
        return KernelHandle(self, name, fn)


def _acquire(entry: _ModuleEntry) -> _ModuleLease:
    with entry.lock:
        entry.lease_count += 1
    return _ModuleLease(entry)


def _release(lease: _ModuleLease) -> None:
    entry = lease._entry
    with entry.lock:
        if not lease._open:
            return
        lease._open = False
        entry.lease_count -= 1
        finalize = entry.retired and entry.lease_count == 0 and entry.handle is not None
    if finalize:
        _sync_and_unload(entry)


def _sync_and_unload(entry: _ModuleEntry) -> None:
    """Synchronize the selected device then unload; errors propagate and
    leave the module handle intact rather than freeing in-flight work."""
    torch = require_torch()
    with torch.cuda.device(entry.device):
        try:
            torch.cuda.synchronize(entry.device)
        except Exception as exc:
            raise GpuError(f"module cleanup synchronization failed on {entry.device}") from exc
        entry.unload()


def _load_module(device, source: bytes,
                 func_names=("ref_init_index", "ref_argsort",
                             "ref_fused_project")) -> _ModuleEntry:
    torch = require_torch()
    vendor = backend_vendor()
    dev = torch.device(device)
    arch = _arch_for(dev)
    if vendor == "amd":
        image = _hiprtc_compile(source, arch)
        driver = _hip_driver()
    else:
        image = _nvrtc_compile(source, arch)
        driver = _cuda_driver()
    if vendor == "amd":
        _proto(driver, _HIP_SPECS, ("hip", id(driver)), "HIP driver")
    else:
        _proto(driver, _CU_SPECS, ("cu", id(driver)), "CUDA driver")
    mod = ctypes.c_void_p()
    image_buf = ctypes.create_string_buffer(image)
    entry = _ModuleEntry(vendor, dev, None, {}, driver)
    with torch.cuda.device(dev):
        try:
            if vendor == "amd":
                rc = driver.hipModuleLoadData(ctypes.byref(mod), image_buf)
            else:
                rc = driver.cuModuleLoadData(ctypes.byref(mod), image_buf)
            if rc != 0:
                raise GpuError(f"module load failed: {rc}")
            entry.handle = mod
            for fname in func_names:
                fn = ctypes.c_void_p()
                if vendor == "amd":
                    rc = driver.hipModuleGetFunction(ctypes.byref(fn), mod, fname.encode())
                else:
                    rc = driver.cuModuleGetFunction(ctypes.byref(fn), mod, fname.encode())
                if rc != 0:
                    raise GpuError(f"module get function {fname} failed: {rc}")
                entry.funcs[fname] = fn
        except Exception as exc:
            try:
                entry.unload()
            except Exception as cleanup:
                exc.add_note(f"module load cleanup failed: {cleanup}")
            raise
    return entry


def _retire(entry: _ModuleEntry) -> bool:
    with entry.lock:
        entry.retired = True
        return entry.lease_count == 0 and entry.handle is not None


def _retire_entries(entries) -> None:
    error = None
    for entry in entries:
        try:
            if _retire(entry):
                _sync_and_unload(entry)
        except Exception as exc:
            if error is None:
                error = exc
            else:
                error.add_note(str(exc))
    if error is not None:
        raise error


def get_module(device, source: bytes, name: str = "reference_sort.cu",
               func_names=("ref_init_index", "ref_argsort",
                           "ref_fused_project")):
    """Compile/cache a module keyed by vendor/device/arch/source hash.

    Returns a context-managed _ModuleLease; one RLock serializes
    lookup+compile+load+insert so concurrent misses cannot duplicate a
    load. Retirement cleanup happens after the lock with every evicted
    entry collected, not only the last.
    """
    torch = require_torch()
    vendor = backend_vendor()
    dev = torch.device(device)
    arch = _arch_for(dev)
    key = (vendor, dev.index, arch, hashlib.sha256(source).hexdigest(),
           func_names)
    evicted: list = []
    with _cache_lock:
        entry = _module_cache.get(key)
        if entry is None:
            entry = _load_module(dev, source, func_names)
            _module_cache[key] = entry
        _module_cache.move_to_end(key)
        while len(_module_cache) > _CACHE_LIMIT:
            _, old = _module_cache.popitem(last=False)
            evicted.append(old)
        lease = _acquire(entry)
    try:
        _retire_entries(evicted)
    except Exception:
        _release(lease)
        raise
    return lease


def clear_cache() -> None:
    """Retire every cached entry and unload the ones with no open lease."""
    with _cache_lock:
        entries = list(_module_cache.values())
        _module_cache.clear()
    _retire_entries(entries)


def launch(device, func, grid: int, block: int, args: list, stream=None,
           keepalive: tuple = ()):
    """Launch a compiled kernel on torch's current stream for ``device``.

    ``args`` is a list of (kind, value) where kind is 'ptr' (device address)
    or 'i64'/'i32' scalars. ``keepalive`` tensors get ``record_stream``
    protection so caching-allocator memory outlives the async launch.
    """
    torch = require_torch()
    if not isinstance(func, KernelHandle):
        raise GpuError("launch requires a KernelHandle from an open module lease")
    dev = torch.device(device)
    entry = func.lease._entry
    if entry.device != dev:
        raise GpuError(
            f"kernel handle belongs to {entry.device}, cannot launch on {dev}"
        )
    vendor = backend_vendor()
    driver = entry.driver
    if vendor == "amd":
        _proto(driver, _HIP_SPECS, ("hip", id(driver)), "HIP driver")
    else:
        _proto(driver, _CU_SPECS, ("cu", id(driver)), "CUDA driver")
    holders = []
    for kind, value in args:
        if kind == "ptr":
            holders.append(ctypes.c_void_p(int(value)))
        elif kind == "i64":
            holders.append(ctypes.c_longlong(int(value)))
        elif kind == "i32":
            holders.append(ctypes.c_int(int(value)))
        else:
            raise GpuError(f"unknown kernel arg kind {kind!r}")
    params = (ctypes.c_void_p * len(holders))()
    for i, h in enumerate(holders):
        params[i] = ctypes.addressof(h)
    if stream is None:
        stream = torch.cuda.current_stream(dev).cuda_stream
    with entry.lock:
        if not func.lease._open:
            raise GpuError(f"kernel handle {func.name!r} lease is closed")
        with torch.cuda.device(dev):
            if vendor == "amd":
                rc = driver.hipModuleLaunchKernel(
                    func.fn, grid, 1, 1, block, 1, 1, 0,
                    ctypes.c_void_p(int(stream)), params, None,
                )
            else:
                rc = driver.cuLaunchKernel(
                    func.fn, grid, 1, 1, block, 1, 1, 0,
                    ctypes.c_void_p(int(stream)), params, None,
                )
            cur = torch.cuda.current_stream(dev)
            for t in keepalive:
                if isinstance(t, torch.Tensor):
                    t.record_stream(cur)
    if rc != 0:
        raise GpuError(f"kernel launch failed: {rc}")
