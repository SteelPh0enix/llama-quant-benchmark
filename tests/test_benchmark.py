"""Tests for benchmark functionality using llama-bench."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llama_quant_bench import (
    convert_hf_to_gguf,
    run_benchmark,
    run_benchmark_all_tests,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@pytest.mark.slow
class TestBenchmark:
    """Tests for benchmark functionality using llama-bench.

    These tests are marked as slow because they involve running
    actual benchmarks on quantized models.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_binaries(self, request: "FixtureRequest") -> None:
        """Skip tests if required binaries are not available."""
        binaries_available = request.getfixturevalue("binaries_available")
        if not binaries_available:
            pytest.skip("Required binaries not available")

    @pytest.mark.slow
    def test_run_benchmark_single(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test running llama-bench on a single test.

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        # First convert the model
        base_gguf = temp_output_dir / "base.gguf"
        convert_hf_to_gguf(str(test_model_path), str(base_gguf))

        # Run benchmark with single test
        output = run_benchmark(str(base_gguf), test_prompt=512)

        # Verify output contains expected content
        assert "pp512" in output or "bench" in output.lower(), (
            "Benchmark output should contain test information"
        )

    @pytest.mark.slow
    def test_run_benchmark_all_tests(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test running llama-bench with all configured tests.

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        # First convert the model
        base_gguf = temp_output_dir / "base.gguf"
        convert_hf_to_gguf(str(test_model_path), str(base_gguf))

        # Run benchmark with all tests
        output = run_benchmark_all_tests(str(base_gguf))

        # Verify output contains test information
        assert "pp" in output or "tg" in output or "bench" in output.lower(), (
            "Benchmark output should contain test information"
        )
