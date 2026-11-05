from tools.gpu_validation.reference import gpu_and_harness_sources, source_paths


def test_additive_gpu_launcher_is_not_part_of_cpu_baseline(tmp_path):
    cpu = tmp_path / "isp_pipeline.py"
    gpu = tmp_path / "isp_pipeline_gpu_batch.py"
    cpu.write_text("cpu")
    gpu.write_text("gpu")
    paths = source_paths(tmp_path)
    assert cpu in paths and gpu not in paths
    identities = gpu_and_harness_sources(tmp_path)
    assert str(gpu.relative_to(tmp_path)) in identities
    assert str(cpu.relative_to(tmp_path)) not in identities
    fixtures = tmp_path / "tests" / "gpu" / "fixtures"
    fixtures.mkdir(parents=True)
    golden = fixtures / "golden.npz"
    golden.write_bytes(b"frozen expected arrays")
    assert "tests/gpu/fixtures/golden.npz" in gpu_and_harness_sources(tmp_path)
