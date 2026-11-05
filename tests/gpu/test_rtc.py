"""RTC loader safety: mocked HIPRTC/CUDA driver coverage.

These are mocked-library tests — they validate ctypes plumbing, cleanup
and lifecycle only, and are NOT evidence of working ROCm execution. Real
CUDA execution is covered by test_reference_sort/test_statistics on
cuda_device.
"""

from __future__ import annotations

import ctypes
import threading

import numpy as np
import pytest

from brilliantisp_gpu.types import GpuError, GpuUnavailableError
from brilliantisp_gpu.ops import rtc


@pytest.fixture(autouse=True)
def isolated_rtc_cache(monkeypatch):
    rtc.clear_cache()
    yield
    rtc.clear_cache()


class _FakeNvrtc:
    def __init__(self, compile_rc=0, code=b"\x00\xff\x10ptx\x00\x00",
                 log_rcs=None, ptx_rcs=None, destroy_rc=0):
        self.compile_rc = compile_rc
        self.code = code
        self.destroyed = 0
        self.opts_seen = None
        self.log_size_rc = 0 if log_rcs is None else log_rcs[0]
        self.log_rc = 0 if log_rcs is None else log_rcs[1]
        self.ptx_size_rc = 0 if ptx_rcs is None else ptx_rcs[0]
        self.ptx_rc = 0 if ptx_rcs is None else ptx_rcs[1]
        self.destroy_rc = destroy_rc

    def nvrtcCreateProgram(self, prog, src, name, nh, headers, names):
        prog._obj.value = 0xABC
        return 0

    def nvrtcCompileProgram(self, prog, nopts, opts):
        self.opts_seen = [opts[i] for i in range(nopts)]
        return self.compile_rc

    def nvrtcGetProgramLogSize(self, prog, size):
        if self.log_size_rc:
            return self.log_size_rc
        size._obj.value = 4
        return 0

    def nvrtcGetProgramLog(self, prog, buf):
        if self.log_rc:
            return self.log_rc
        ctypes.memmove(buf, b"LOG\x00", 4)
        return 0

    def nvrtcGetPTXSize(self, prog, size):
        if self.ptx_size_rc:
            return self.ptx_size_rc
        size._obj.value = len(self.code)
        return 0

    def nvrtcGetPTX(self, prog, buf):
        if self.ptx_rc:
            return self.ptx_rc
        ctypes.memmove(buf, self.code, len(self.code))
        return 0

    def nvrtcDestroyProgram(self, prog):
        self.destroyed += 1
        return self.destroy_rc


class _FakeCu:
    def __init__(self, load_rc=0, func_rc=0, unload_rc=0):
        self.load_rc = load_rc
        self.func_rc = func_rc
        self.unload_rc = unload_rc
        self.loaded = 0
        self.unloaded = []
        self.launched = []

    def cuModuleLoadData(self, mod, image):
        self.loaded += 1
        mod._obj.value = 0x1000 + self.loaded
        return self.load_rc

    def cuModuleGetFunction(self, fn, mod, name):
        if self.func_rc:
            return self.func_rc
        fn._obj.value = 0x5000
        return 0

    def cuModuleUnload(self, mod):
        self.unloaded.append(mod)
        return self.unload_rc

    def cuLaunchKernel(self, fn, *args):
        self.launched.append(fn)
        return 0


def _patch(monkeypatch, nvrtc=None, cu=None):
    if nvrtc is not None:
        monkeypatch.setattr(rtc, "_cuda_compiler", lambda: nvrtc)
    if cu is not None:
        monkeypatch.setattr(rtc, "_cuda_driver", lambda: cu)
    rtc._proto_done.clear()


def test_nvrtc_ptx_returns_full_bytes(monkeypatch):
    fake = _FakeNvrtc(code=b"\x00binary\x00with\x00nuls\x00end")
    monkeypatch.setattr(rtc, "_cuda_compiler", lambda: fake)
    rtc._proto_done.clear()
    image = rtc._nvrtc_compile(b"x", "89")
    assert image == fake.code
    assert fake.destroyed == 1
    assert any(b"compute_89" in o for o in fake.opts_seen)
    assert any(b"fmad" in o for o in fake.opts_seen)


def test_nvrtc_compile_failure_cleans_program(monkeypatch):
    fake = _FakeNvrtc(compile_rc=5)
    monkeypatch.setattr(rtc, "_cuda_compiler", lambda: fake)
    rtc._proto_done.clear()
    with pytest.raises(GpuError, match="compile failed"):
        rtc._nvrtc_compile(b"broken", "89")
    assert fake.destroyed == 1


def test_nvrtc_cleanup_rc_preserved(monkeypatch):
    fake = _FakeNvrtc(compile_rc=5, destroy_rc=9)
    monkeypatch.setattr(rtc, "_cuda_compiler", lambda: fake)
    rtc._proto_done.clear()
    with pytest.raises(GpuError) as ei:
        rtc._nvrtc_compile(b"broken", "89")
    assert "compile failed" in str(ei.value)
    assert fake.destroyed == 1
    assert any("DestroyProgram failed: 9" in n for n in ei.value.__notes__)


@pytest.mark.parametrize("backend", ["nvrtc", "hiprtc"])
@pytest.mark.parametrize("fault,expected", [
    ({"log_rcs": (3, 0)}, "GetProgramLogSize"),
    ({"log_rcs": (0, 3)}, "GetProgramLog"),
    ({"ptx_rcs": (4, 0)}, "Size"),
    ({"ptx_rcs": (0, 4)}, "Get"),
    ({"destroy_rc": 9}, "DestroyProgram"),
])
def test_compiler_api_failures_checked(monkeypatch, backend, fault, expected):
    fake = _FakeNvrtc(**fault)
    if backend == "hiprtc":
        for name in rtc._NVRTC_SPECS:
            setattr(fake, name.replace("nvrtc", "hiprtc").replace("PTX", "Code"),
                    getattr(fake, name))
        monkeypatch.setattr(rtc, "_hip_compiler", lambda: fake)
        compile_source, arch = rtc._hiprtc_compile, "gfx1030"
    else:
        monkeypatch.setattr(rtc, "_cuda_compiler", lambda: fake)
        compile_source, arch = rtc._nvrtc_compile, "89"
    with pytest.raises(GpuError, match=expected):
        compile_source(b"src", arch)
    assert fake.destroyed == 1


def test_destroy_failure_does_not_use_callers_exception(monkeypatch):
    fake = _FakeNvrtc(destroy_rc=9)
    monkeypatch.setattr(rtc, "_cuda_compiler", lambda: fake)
    try:
        raise LookupError("caller")
    except LookupError:
        with pytest.raises(GpuError, match="DestroyProgram"):
            rtc._nvrtc_compile(b"src", "89")


def test_module_load_failure_does_not_unload_null_handle(monkeypatch, cuda_device):
    rtc.clear_cache()
    fake_cu = _FakeCu(load_rc=7)
    _patch(monkeypatch, _FakeNvrtc(), fake_cu)
    with pytest.raises(GpuError, match="module load"):
        rtc.get_module(cuda_device, rtc.load_source("reference_sort.cu"))
    assert fake_cu.loaded == 1 and not fake_cu.unloaded


def test_module_unload_on_get_function_failure(monkeypatch, cuda_device):
    fake_rtc = _FakeNvrtc()
    fake_cu = _FakeCu(func_rc=7)
    _patch(monkeypatch, fake_rtc, fake_cu)
    rtc.clear_cache()
    with pytest.raises(GpuError):
        rtc.get_module(cuda_device, rtc.load_source("reference_sort.cu"))
    assert len(fake_cu.unloaded) == 1


def test_module_cache_eviction_unloads(monkeypatch, cuda_device):
    fake_rtc = _FakeNvrtc()
    fake_cu = _FakeCu()
    _patch(monkeypatch, fake_rtc, fake_cu)
    monkeypatch.setattr(rtc, "_CACHE_LIMIT", 3)
    rtc.clear_cache()
    src = rtc.load_source("reference_sort.cu")
    for i in range(5):
        lease = rtc.get_module(cuda_device, src + b" " * (i + 1))
        lease.__exit__(None, None, None)
    assert len(rtc._module_cache) == 3
    assert len(fake_cu.unloaded) == 2


def test_lease_retained_across_eviction(monkeypatch, cuda_device):
    """Lease A stays usable while A is open; unload once after A exits."""
    fake_rtc = _FakeNvrtc()
    fake_cu = _FakeCu()
    _patch(monkeypatch, fake_rtc, fake_cu)
    monkeypatch.setattr(rtc, "_CACHE_LIMIT", 1)
    rtc.clear_cache()
    src_a = rtc.load_source("reference_sort.cu")
    src_b = src_a + b" "
    lease_a = rtc.get_module(cuda_device, src_a)
    handle_a = lease_a.funcs("ref_argsort")
    lease_b = rtc.get_module(cuda_device, src_b)
    lease_b.__exit__(None, None, None)
    assert len(fake_cu.unloaded) == 0
    with pytest.raises(GpuError):
        lease_b.funcs("ref_argsort")
    rtc.launch(cuda_device, handle_a, 1, 1,
               [("ptr", 0), ("ptr", 0), ("i64", 1), ("i32", 0)])
    assert len(fake_cu.launched) == 1
    lease_a.__exit__(None, None, None)
    assert len(fake_cu.unloaded) == 1
    lease_a.__exit__(None, None, None)
    assert lease_a._entry.lease_count == 0 and len(fake_cu.unloaded) == 1
    with pytest.raises(GpuError, match="closed"):
        lease_a.__enter__()
    with pytest.raises(GpuError):
        rtc.launch(cuda_device, handle_a, 1, 1,
                   [("ptr", 0), ("ptr", 0), ("i64", 1), ("i32", 0)])


def test_concurrent_load_same_key_single_module(monkeypatch, cuda_device):
    fake_rtc = _FakeNvrtc()
    fake_cu = _FakeCu()
    _patch(monkeypatch, fake_rtc, fake_cu)
    rtc.clear_cache()
    src = rtc.load_source("reference_sort.cu")
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            lease = rtc.get_module(cuda_device, src)
            results.append(lease)
            lease.__exit__(None, None, None)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    assert len(results) == 2
    assert fake_cu.loaded == 1
    assert len(rtc._module_cache) == 1


def test_module_get_function_under_selected_device(monkeypatch, cuda_device):
    calls = []

    class DevAware(_FakeCu):
        def cuModuleGetFunction(self, fn, mod, name):
            torch = __import__("torch")
            calls.append(torch.cuda.current_device())
            return super().cuModuleGetFunction(fn, mod, name)

    fake_rtc = _FakeNvrtc()
    fake_cu = DevAware()
    _patch(monkeypatch, fake_rtc, fake_cu)
    rtc.clear_cache()
    lease = rtc.get_module(cuda_device, rtc.load_source("reference_sort.cu"))
    lease.__exit__(None, None, None)
    assert calls and all(c == cuda_device.index for c in calls)


def test_sync_failure_blocks_unload(monkeypatch, cuda_device):
    fake_rtc = _FakeNvrtc()
    fake_cu = _FakeCu()
    _patch(monkeypatch, fake_rtc, fake_cu)
    monkeypatch.setattr(rtc, "_CACHE_LIMIT", 1)
    rtc.clear_cache()
    torch = pytest.importorskip("torch")
    src_a = rtc.load_source("reference_sort.cu")
    lease_a = rtc.get_module(cuda_device, src_a)
    lease_a.__exit__(None, None, None)
    original_sync = torch.cuda.synchronize
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sync boom")))
    with pytest.raises(GpuError, match="synchronization"):
        rtc.get_module(cuda_device, src_a + b" ")
    assert len(fake_cu.unloaded) == 0
    assert lease_a._entry.handle is not None
    assert all(entry.lease_count == 0 for entry in rtc._module_cache.values())
    monkeypatch.setattr(torch.cuda, "synchronize", original_sync)
    rtc._sync_and_unload(lease_a._entry)
    rtc.clear_cache()


def test_unload_failure_named_error(monkeypatch, cuda_device):
    fake_rtc = _FakeNvrtc()
    fake_cu = _FakeCu(unload_rc=4)
    _patch(monkeypatch, fake_rtc, fake_cu)
    monkeypatch.setattr(rtc, "_CACHE_LIMIT", 1)
    rtc.clear_cache()
    src_a = rtc.load_source("reference_sort.cu")
    lease_a = rtc.get_module(cuda_device, src_a)
    lease_a.__exit__(None, None, None)
    with pytest.raises(GpuError, match="unload"):
        rtc.get_module(cuda_device, src_a + b" ")
    fake_cu.unload_rc = 0
    rtc._sync_and_unload(lease_a._entry)
    rtc.clear_cache()


def test_hiprtc_code_preserves_nul_bytes():
    class FakeHiprtc:
        def hiprtcCreateProgram(self, prog, src, name, nh, headers, names):
            prog._obj.value = 0xDEF
            return 0

        def hiprtcCompileProgram(self, prog, nopts, opts):
            return 0

        def hiprtcGetProgramLogSize(self, prog, size):
            size._obj.value = 1
            return 0

        def hiprtcGetProgramLog(self, prog, buf):
            return 0

        def hiprtcGetCodeSize(self, prog, size):
            size._obj.value = len(self.blob)
            return 0

        def hiprtcGetCode(self, prog, buf):
            ctypes.memmove(buf, self.blob, len(self.blob))
            return 0

        def hiprtcDestroyProgram(self, prog):
            return 0

        blob = b"\x7fELF\x00\x01\x02\x00\x03machine\x00code"

    rtc._libs.pop("HIPRTC", None)
    try:
        rtc._libs["HIPRTC"] = FakeHiprtc()
        rtc._proto_done.discard(("hiprtc", id(FakeHiprtc)))
        image = rtc._hiprtc_compile(b"src", "gfx1030")
        assert image == FakeHiprtc.blob
    finally:
        rtc._libs.pop("HIPRTC", None)


def test_hip_missing_gcn_arch_named_error(cuda_device, monkeypatch):
    """On this NVIDIA build the property exists but is not a gfx arch —
    the HIP path must still fail with a named GpuUnavailableError."""
    monkeypatch.setattr(rtc, "backend_vendor", lambda: "amd")
    with pytest.raises(GpuUnavailableError, match="gcnArchName"):
        rtc._arch_for(cuda_device)


def test_hip_arch_accepted_shape(cuda_device, monkeypatch):
    monkeypatch.setattr(rtc, "backend_vendor", lambda: "amd")

    class _Props:
        major = 10
        minor = 3
        gcnArchName = "gfx1030"

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda d: _Props()
    )
    assert rtc._arch_for(cuda_device) == "gfx1030"


def test_fused_project_rejects_cross_device(cuda_device):
    torch = pytest.importorskip("torch")
    from brilliantisp_gpu.ops.reference_sort import fused_project
    rows = torch.rand(4, 3, dtype=torch.float64, device=cuda_device)
    unit_cpu = torch.rand(3, dtype=torch.float64)
    with pytest.raises(GpuError):
        fused_project(rows, unit_cpu)
    with pytest.raises(GpuError):
        fused_project(rows.float(), torch.rand(3, device=cuda_device))
    with pytest.raises(GpuError):
        fused_project(torch.rand(4, 3, dtype=torch.float64, device=cuda_device),
                      torch.rand(4, dtype=torch.float64, device=cuda_device))


def test_fused_project_nondefault_stream_allocator_churn(cuda_device):
    torch = pytest.importorskip("torch")
    from brilliantisp_gpu.ops.reference_sort import fused_project
    rows_np = np.random.RandomState(5).rand(64, 3)
    s = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(s):
        rows = torch.as_tensor(rows_np, dtype=torch.float64, device=cuda_device)
        unit = torch.ones(3, dtype=torch.float64, device=cuda_device) / (3 ** 0.5)
        out = fused_project(rows, unit)
        churn = [torch.rand(4096, device=cuda_device) for _ in range(8)]
        del churn
        result = out.cpu().numpy()
    expected = rows_np @ (np.ones(3) / np.sqrt(3))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0.0)


def test_module_func_names_default_and_custom(monkeypatch, cuda_device):
    """Default 3-name ABI and custom func_names tuple coexist in cache."""
    requested = []

    class RecordingCu(_FakeCu):
        def cuModuleGetFunction(self, fn, mod, name):
            requested.append(name.decode())
            return super().cuModuleGetFunction(fn, mod, name)

    fake_rtc = _FakeNvrtc()
    fake_cu = RecordingCu()
    _patch(monkeypatch, fake_rtc, fake_cu)
    rtc.clear_cache()
    src = rtc.load_source("reference_sort.cu")
    lease = rtc.get_module(cuda_device, src)
    lease.__exit__(None, None, None)
    lease2 = rtc.get_module(
        cuda_device, src,
        func_names=("area_f32_fast", "area_f32_generic"))
    lease2.__exit__(None, None, None)
    assert fake_cu.loaded == 2
    assert len(rtc._module_cache) == 2
    assert requested == ["ref_init_index", "ref_argsort", "ref_fused_project",
                         "area_f32_fast", "area_f32_generic"]
    lease3 = rtc.get_module(
        cuda_device, src,
        func_names=("area_f32_fast", "area_f32_generic"))
    lease3.__exit__(None, None, None)
    assert fake_cu.loaded == 2 and len(requested) == 5
