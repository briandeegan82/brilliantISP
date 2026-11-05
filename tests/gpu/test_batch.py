from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, get_ident

import numpy as np
import pytest

from brilliantisp_gpu.batch import process_batch
from brilliantisp_gpu.types import BatchItem, GpuOptions
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingOptions, ProcessingState
from test_microbatch import CAPTURES, _assert_same, _config, _sources


REPO = Path(__file__).resolve().parents[2]


def _options():
    return ProcessingOptions(run_id="batch_reference", capture_stages=CAPTURES)


def _expected(item, options, state=None):
    opts = dataclasses.replace(
        options, run_id=f"{options.run_id}:{item.item_id}",
        output_dir=item.output_dir if item.output_dir is not None else options.output_dir,
    )
    return process_frame(item.source, item.config, state=state, options=opts)


@pytest.mark.parametrize("maximum", (1, 2, 4))
def test_batch_independent_results_and_actual_sizes(cuda_device, maximum):
    import brilliantisp_gpu.batch as batch

    cfg, opts = _config(), _options()
    sources = _sources(6)
    items = [BatchItem(f"item_{i}", source, cfg) for i, source in enumerate(sources)]
    expected = [_expected(item, opts) for item in items]
    before = [source.copy() for source in sources]
    observed = []
    original = batch.execute_group

    def execute(engines, decoded):
        observed.append(len(engines))
        return original(engines, decoded)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "execute_group", execute)
        outcomes = list(process_batch(
            iter(items), options=opts,
            gpu=GpuOptions(device=str(cuda_device), max_batch_size=maximum),
        ))
    assert [o.item_id for o in outcomes] == [i.item_id for i in items]
    assert observed == [maximum] * (6 // maximum) + ([6 % maximum] if 6 % maximum else [])
    assert [o.batch_size for o in outcomes] == [
        n for n in observed for _ in range(n)
    ]
    for outcome, cpu in zip(outcomes, expected):
        assert outcome.status == "success" and outcome.error is None
        assert outcome.device["vendor"] == "nvidia"
        assert outcome.metrics["compute_transfer_group_ns"] >= 0
        assert outcome.attempts and outcome.attempts[-1].status == "success"
        _assert_same(cpu, outcome.result)
    for source, reference in zip(sources, before):
        np.testing.assert_array_equal(source, reference)
    assert opts == _options()


def test_batch_mixed_configs_and_geometry_are_never_padded(cuda_device):
    cfg_a, cfg_b = _config(), _config()
    cfg_b["demosaic"]["algorithm"] = "bilinear"
    sources = _sources(4)
    cfg_c = _config()
    cfg_c["sensor_info"].update(height=7, width=9)
    sources[3] = sources[3][:7, :9].copy()
    configs = [cfg_a, cfg_b, cfg_b, cfg_c]
    opts = _options()
    items = [BatchItem(f"mixed_{i}", s, c) for i, (s, c) in enumerate(zip(sources, configs))]
    expected = [_expected(item, opts) for item in items]
    outcomes = list(process_batch(items, options=opts, gpu=GpuOptions(device=str(cuda_device))))
    assert [o.batch_size for o in outcomes] == [1, 2, 2, 1]
    for outcome, cpu in zip(outcomes, expected):
        assert outcome.status == "success"
        _assert_same(cpu, outcome.result)


@pytest.mark.parametrize("prefetch", (0, 1, 2))
def test_batch_prefetch_bounds_and_main_thread_acceptance(cuda_device, prefetch):
    accepted, main = [], get_ident()
    cfg = _config()

    def stream():
        for i in range(20):
            assert get_ident() == main
            accepted.append(i)
            yield BatchItem(f"bounded_{i}", _sources(2)[1], cfg)

    iterator = process_batch(
        stream(), gpu=GpuOptions(device=str(cuda_device), prefetch=prefetch),
    )
    try:
        outcome = next(iterator)
        assert outcome.status == "success"
        assert len(accepted) <= (prefetch + 1) * 4
    finally:
        iterator.close()


@pytest.mark.parametrize("prefetch", (0, 1, 2))
def test_batch_bounds_hold_after_memory_reduces_every_group(
    cuda_device, monkeypatch, prefetch,
):
    import torch

    accepted = []
    cfg = _config()
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *args: (8 * 1024**2, 16 * 1024**2),
    )

    def stream():
        for i in range(20):
            accepted.append(i)
            yield BatchItem(f"reduced_{i}", _sources(2)[1], cfg)

    iterator = process_batch(
        stream(), gpu=GpuOptions(device=str(cuda_device), prefetch=prefetch),
    )
    completed = 0
    try:
        for outcome in iterator:
            assert outcome.status == "success" and outcome.batch_size == 1
            assert len(accepted) - completed <= (prefetch + 1) * 4
            completed += 1
    finally:
        iterator.close()
    assert completed == 20


def test_batch_freezes_accepted_arrays_and_config_before_next_pull(cuda_device):
    cfg = _config()
    sources = _sources(2)
    reference_cfg, reference_sources = copy.deepcopy(cfg), [a.copy() for a in sources]
    opts = _options()
    original_items = [
        BatchItem(f"frozen_{i}", source, reference_cfg)
        for i, source in enumerate(reference_sources)
    ]
    expected = [_expected(item, opts) for item in original_items]

    def stream():
        yield BatchItem("frozen_0", sources[0], cfg)
        sources[0].fill(4095)
        cfg["demosaic"]["algorithm"] = "bilinear"
        cfg["demosaic"]["algorithm"] = "malvar"
        yield BatchItem("frozen_1", sources[1], cfg)
        sources[1].fill(0)
        cfg["demosaic"]["algorithm"] = "bilinear"

    outcomes = list(process_batch(
        stream(), options=opts, gpu=GpuOptions(device=str(cuda_device)),
    ))
    for outcome, cpu in zip(outcomes, expected):
        assert outcome.status == "success"
        _assert_same(cpu, outcome.result)


@pytest.mark.parametrize("split", (False, True))
def test_batch_oom_split_computes_all_before_any_write(
    cuda_device, monkeypatch, tmp_path, split,
):
    import torch
    import brilliantisp_gpu.batch as batch

    cfg = _config()
    items = [
        BatchItem(f"oom_{i}", source, cfg, output_dir=tmp_path / f"out_{i}",
                  output_stem=f"item_{i}")
        for i, source in enumerate(_sources(4))
    ]
    opts = ProcessingOptions(run_id="oom", save_outputs=True)
    original_execute, original_commit = batch.execute_group, batch.commit_frame
    computed, sizes = set(), []

    def execute(engines, decoded):
        sizes.append(len(engines))
        if len(engines) > 1:
            assert not any(tmp_path.iterdir())
            raise torch.cuda.OutOfMemoryError("injected group allocation failure")
        result = original_execute(engines, decoded)
        computed.add(engines[0].run_id)
        return result

    def commit(pending):
        assert computed == {f"oom:{item.item_id}" for item in items}
        return original_commit(pending)

    monkeypatch.setattr(batch, "execute_group", execute)
    monkeypatch.setattr(batch, "commit_frame", commit)
    outcomes = list(process_batch(
        items, options=opts,
        gpu=GpuOptions(device=str(cuda_device), split_on_oom=split),
    ))
    if split:
        assert sizes == [4, 2, 1, 1, 2, 1, 1]
        assert all(o.status == "success" and o.batch_size == 1 for o in outcomes)
        assert all(any(a.error is not None and a.error.kind == "gpu_oom"
                       for a in o.attempts) for o in outcomes)
        paths = [p for o in outcomes for p in o.result.artifacts]
        assert len(paths) == len(set(paths))
        assert set(paths) == {p for p in tmp_path.rglob("*") if p.is_file()}
    else:
        assert sizes == [4]
        assert all(o.status == "failed" and o.error.kind == "gpu_oom" for o in outcomes)
        assert all(o.result is None for o in outcomes)
        assert not any(tmp_path.iterdir())


def test_batch_non_oom_failure_isolated_without_hiding_cause(cuda_device, monkeypatch):
    import brilliantisp_gpu.batch as batch

    cfg, opts = _config(), _options()
    items = [BatchItem(name, source, cfg) for name, source in zip(
        ("good_0", "bad", "good_1"), _sources(3),
    )]
    original = batch.execute_group

    def execute(engines, decoded):
        if any(e.run_id.endswith(":bad") for e in engines):
            raise RuntimeError("injected processing failure")
        return original(engines, decoded)

    monkeypatch.setattr(batch, "execute_group", execute)
    outcomes = list(process_batch(items, options=opts, gpu=GpuOptions(device=str(cuda_device))))
    assert [o.status for o in outcomes] == ["success", "failed", "success"]
    assert outcomes[1].error.exception_type == "RuntimeError"
    assert outcomes[1].error.kind == "processing"
    assert "injected processing failure" in outcomes[1].error.message
    for i in (0, 2):
        _assert_same(_expected(items[i], opts), outcomes[i].result)
        assert any(a.status == "failed" for a in outcomes[i].attempts)


def test_batch_interleaved_sequences_and_failure_blocking(cuda_device, monkeypatch):
    import brilliantisp_gpu.batch as batch

    cfg = _config()
    cfg["digital_gain"].update(
        is_enable=True, is_auto=True, gain_array=[1.0, 2.0, 4.0], current_gain=0,
    )
    cfg["auto_exposure"].update(
        is_enable=True, exposure_correction_mode="direct",
        rerun_from_digital_gain=True, target_luminance=.5,
    )
    cfg["auto_white_balance"].update(
        is_enable=True, algorithm="pca", percentage=20,
        underexposed_percentage=0.0, overexposed_percentage=0.0,
    )
    cfg["white_balance"].update(is_enable=True, is_auto=True)
    source = _sources(4)
    items = [
        BatchItem("a0", source[3], cfg, sequence_id="a", frame_index=0),
        BatchItem("plain0", source[1], cfg),
        BatchItem("a1", source[2], cfg, sequence_id="a", frame_index=1),
        BatchItem("b0", source[1], cfg, sequence_id="b", frame_index=0),
        BatchItem("a2", source[0], cfg, sequence_id="a", frame_index=2),
        BatchItem("b1", source[2], cfg, sequence_id="b", frame_index=1),
        BatchItem("plain1", source[3], cfg),
    ]
    opts = _options()
    states = {"a": ProcessingState(), "b": ProcessingState()}
    expected = {}
    for item in items:
        if item.item_id not in ("a1", "a2"):
            expected[item.item_id] = _expected(item, opts, states.get(item.sequence_id))
    original = batch.execute_group

    def execute(engines, decoded):
        if any(e.run_id.endswith(":a1") for e in engines):
            raise RuntimeError("sequence frame failed")
        return original(engines, decoded)

    monkeypatch.setattr(batch, "execute_group", execute)
    outcomes = list(process_batch(items, options=opts, gpu=GpuOptions(device=str(cuda_device))))
    assert [o.item_id for o in outcomes] == [i.item_id for i in items]
    assert [o.status for o in outcomes] == [
        "success", "success", "failed", "success", "blocked", "success", "success",
    ]
    for item, outcome in zip(items, outcomes):
        if outcome.status == "success":
            _assert_same(expected[item.item_id], outcome.result)
            if item.sequence_id:
                assert outcome.batch_size == 1
        elif outcome.status == "blocked":
            assert outcome.result is None and outcome.error.kind == "blocked"


def test_batch_input_failure_and_cancellation_do_not_write(cuda_device, tmp_path):
    cfg = _config()
    items = [
        BatchItem("missing", tmp_path / "missing.raw", cfg),
        BatchItem("valid", _sources(2)[1], cfg),
    ]
    outcomes = list(process_batch(items, gpu=GpuOptions(device=str(cuda_device))))
    assert [o.status for o in outcomes] == ["failed", "success"]
    assert outcomes[0].error.kind == "input"
    token = Event()
    token.set()
    outcomes = list(process_batch(
        items[1:], gpu=GpuOptions(device=str(cuda_device)),
        options=ProcessingOptions(cancellation_token=token, save_outputs=True, output_dir=tmp_path / "out"),
    ))
    assert [o.status for o in outcomes] == ["cancelled"]
    assert not (tmp_path / "out").exists()


def test_batch_cancellation_after_gpu_output_prevents_commits(cuda_device, tmp_path):
    cfg, token = _config(), Event()
    items = [BatchItem(f"cancel_{i}", source, cfg) for i, source in enumerate(_sources(2))]

    def callback(event):
        if event.stage_id == "output" and event.status == "completed":
            token.set()

    outcomes = list(process_batch(
        items, gpu=GpuOptions(device=str(cuda_device)),
        options=ProcessingOptions(
            cancellation_token=token, progress_callback=callback,
            save_outputs=True, output_dir=tmp_path / "output",
        ),
    ))
    assert [o.status for o in outcomes] == ["cancelled", "cancelled"]
    assert all(o.result is None for o in outcomes)
    assert not (tmp_path / "output").exists()


def test_batch_runtime_unavailable_is_a_structured_outcome(monkeypatch):
    import brilliantisp_gpu.batch as batch
    from brilliantisp_gpu.types import GpuUnavailableError

    def unavailable():
        raise GpuUnavailableError("injected unavailable GPU")

    monkeypatch.setattr(batch, "require_torch", unavailable)
    outcomes = list(process_batch([
        BatchItem(f"unavailable_{i}", source, _config())
        for i, source in enumerate(_sources(2))
    ]))
    assert len(outcomes) == 2
    assert all(o.status == "failed" and o.result is None for o in outcomes)
    assert all(o.error.kind == "gpu_unavailable" for o in outcomes)
    assert all(o.batch_size == 0 for o in outcomes)
    assert all(o.device["vendor"] is None for o in outcomes)


def test_batch_uses_runtime_device_metadata(cuda_device, monkeypatch):
    import brilliantisp_gpu.device as device

    properties = dict(
        vendor="amd", name="metadata propagation stub", torch="test",
        hip="test", cuda=None, total_memory=1,
    )
    monkeypatch.setattr(device, "device_properties", lambda selected: dict(properties))
    outcomes = list(process_batch(
        [BatchItem("metadata", _sources(2)[1], _config())],
        gpu=GpuOptions(device=str(cuda_device)),
    ))
    assert outcomes[0].status == "success"
    assert outcomes[0].device["vendor"] == "amd"
    assert outcomes[0].device["hip"] == "test"


def test_batch_blocked_sequence_takes_precedence_over_prefetched_input_error(
    cuda_device, monkeypatch, tmp_path,
):
    import brilliantisp_gpu.batch as batch

    original = batch.execute_group

    def execute(engines, decoded):
        if engines[0].run_id.endswith(":failed"):
            raise RuntimeError("sequence failed")
        return original(engines, decoded)

    monkeypatch.setattr(batch, "execute_group", execute)
    cfg = _config()
    items = [
        BatchItem("failed", _sources(2)[1], cfg, sequence_id="s", frame_index=0),
        BatchItem("blocked", tmp_path / "missing.raw", cfg, sequence_id="s", frame_index=1),
        BatchItem("independent", _sources(2)[1], cfg),
    ]
    outcomes = list(process_batch(items, gpu=GpuOptions(device=str(cuda_device))))
    assert [o.status for o in outcomes] == ["failed", "blocked", "success"]
    assert outcomes[1].error.kind == "blocked" and outcomes[1].batch_size == 0


def test_batch_oom_cleanup_occurs_after_failed_attempt_tensor_release(
    cuda_device, monkeypatch,
):
    import weakref
    import torch
    import brilliantisp_gpu.batch as batch

    original_execute, original_empty = batch.execute_group, torch.cuda.empty_cache
    references, alive_at_cleanup = [], []

    def execute(engines, decoded):
        if len(engines) > 1:
            temporary = torch.ones((8, 10), device=cuda_device)
            references.append(weakref.ref(temporary))
            raise torch.cuda.OutOfMemoryError("injected live-tensor failure")
        return original_execute(engines, decoded)

    def cleanup():
        alive_at_cleanup.append(any(ref() is not None for ref in references))
        return original_empty()

    monkeypatch.setattr(batch, "execute_group", execute)
    monkeypatch.setattr(torch.cuda, "empty_cache", cleanup)
    outcomes = list(process_batch(
        [BatchItem(f"release_{i}", source, _config()) for i, source in enumerate(_sources(2))],
        gpu=GpuOptions(device=str(cuda_device)),
    ))
    assert all(o.status == "success" for o in outcomes)
    assert alive_at_cleanup and not any(alive_at_cleanup)


def test_batch_pre_compute_sync_failure_is_structured(cuda_device, monkeypatch):
    import torch

    def fail(*args, **kwargs):
        raise RuntimeError("injected pre-compute synchronization failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    outcome, = list(process_batch(
        [BatchItem("sync", _sources(2)[1], _config())],
        gpu=GpuOptions(device=str(cuda_device)),
    ))
    assert outcome.status == "failed"
    assert outcome.error.kind == "processing"
    assert "synchronization failure" in outcome.error.message


def test_batch_commit_failure_is_timed_and_blocks_only_its_sequence(
    cuda_device, monkeypatch,
):
    import brilliantisp_gpu.batch as batch

    original = batch.commit_frame

    def commit(pending):
        if pending.engine.run_id.endswith(":io_failed"):
            raise OSError("injected writer failure")
        return original(pending)

    monkeypatch.setattr(batch, "commit_frame", commit)
    cfg = _config()
    items = [
        BatchItem("io_failed", _sources(2)[1], cfg, sequence_id="s", frame_index=0),
        BatchItem("io_blocked", _sources(2)[1], cfg, sequence_id="s", frame_index=1),
        BatchItem("io_good", _sources(2)[1], cfg),
    ]
    outcomes = list(process_batch(items, gpu=GpuOptions(device=str(cuda_device))))
    assert [o.status for o in outcomes] == ["failed", "blocked", "success"]
    assert outcomes[0].error.kind == "io"
    assert outcomes[0].metrics["commit_ns"] > 0


def test_batch_structural_constraints():
    cfg, source = _config(), _sources(1)[0]
    for kwargs in (
        dict(item_id=""), dict(sequence_id="s"), dict(frame_index=0),
        dict(sequence_id="s", frame_index=True),
        dict(sequence_id="s", frame_index=-1),
        dict(output_stem="../bad"), dict(output_stem=r"C:\bad"),
    ):
        args = dict(item_id="valid", source=source, config=cfg)
        args.update(kwargs)
        with pytest.raises((TypeError, ValueError)):
            BatchItem(**args)
    with pytest.raises(ValueError, match="duplicate"):
        list(process_batch([BatchItem("same", source, cfg), BatchItem("same", source, cfg)]))
    with pytest.raises(ValueError, match="increasing"):
        list(process_batch([
            BatchItem("one", source, cfg, sequence_id="s", frame_index=2),
            BatchItem("two", source, cfg, sequence_id="s", frame_index=1),
        ]))


def test_batch_empty_is_headless_and_torch_optional():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'tkinter', 'PyQt5', 'PyQt6', 'PySide6', 'dearpygui'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from brilliantisp_gpu import process_batch, BatchItem, BatchOutcome
assert list(process_batch(())) == []
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "DISPLAY": "", "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
