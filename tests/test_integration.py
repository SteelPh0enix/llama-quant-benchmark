"""Integration tests for llama-quant-benchmark.

These tests verify the core functionality of the llama-quant-benchmark script,
including model detection, conversion, quantization, and benchmarking.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llama_quant_bench import (
    QuantizationType,
    convert_hf_to_gguf,
    download_converter,
    infer_model_name,
    is_gguf_file,
    is_huggingface_model,
    parse_llama_bench_output,
    quantize_model,
    run_benchmark,
    run_benchmark_all_tests,
    run_full_benchmark,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


# =============================================================================
# MODEL DETECTION TESTS
# =============================================================================


class TestModelDetection:
    """Tests for model type detection functions."""

    def test_is_huggingface_model_valid(self, temp_hf_model_dir: Path) -> None:
        """Test that a valid HF model directory is detected correctly.

        Args:
            temp_hf_model_dir: Fixture providing a mock HF model directory.
        """
        result = is_huggingface_model(str(temp_hf_model_dir))
        assert result is True, "Valid HF model directory should be detected"

    def test_is_huggingface_model_invalid(self, temp_output_dir: Path) -> None:
        """Test that a non-HF directory returns False.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        # Create a directory without HF files
        non_hf_dir = temp_output_dir / "not_a_model"
        non_hf_dir.mkdir()

        result = is_huggingface_model(str(non_hf_dir))
        assert result is False, "Non-HF directory should return False"

    def test_is_huggingface_model_nonexistent(self) -> None:
        """Test that a non-existent path returns False."""
        result = is_huggingface_model("/path/that/does/not/exist")
        assert result is False, "Non-existent path should return False"

    def test_is_huggingface_model_file(self, temp_output_dir: Path) -> None:
        """Test that a file (not directory) returns False.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        # Create a file instead of directory
        test_file = temp_output_dir / "config.json"
        test_file.write_text("{}")

        result = is_huggingface_model(str(test_file))
        assert result is False, "File should not be detected as HF model"

    def test_is_gguf_file_valid(self, temp_gguf_file: Path) -> None:
        """Test that a valid .gguf file is detected.

        Args:
            temp_gguf_file: Fixture providing a mock GGUF file.
        """
        result = is_gguf_file(str(temp_gguf_file))
        assert result is True, "Valid GGUF file should be detected"

    def test_is_gguf_file_invalid(self, temp_output_dir: Path) -> None:
        """Test that a non-.gguf file returns False.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        # Create a file with wrong extension
        wrong_file = temp_output_dir / "model.bin"
        wrong_file.touch()

        result = is_gguf_file(str(wrong_file))
        assert result is False, "Non-GGUF file should return False"

    def test_is_gguf_file_directory(self, temp_output_dir: Path) -> None:
        """Test that a directory returns False.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        result = is_gguf_file(str(temp_output_dir))
        assert result is False, "Directory should not be detected as GGUF file"

    def test_is_gguf_file_nonexistent(self) -> None:
        """Test that a non-existent path returns False."""
        result = is_gguf_file("/path/that/does/not/exist.gguf")
        assert result is False, "Non-existent path should return False"


# =============================================================================
# MODEL NAME INFERENCE TESTS
# =============================================================================


class TestModelNameInference:
    """Tests for model name inference from paths."""

    def test_infer_model_name_from_hf_directory(self, temp_hf_model_dir: Path) -> None:
        """Test extracting model name from HF directory path.

        Args:
            temp_hf_model_dir: Fixture providing a mock HF model directory.
        """
        result = infer_model_name(str(temp_hf_model_dir))
        assert result == "mock_hf_model", f"Expected 'mock_hf_model', got '{result}'"

    def test_infer_model_name_from_gguf_path(self, temp_gguf_file: Path) -> None:
        """Test extracting model name from GGUF file path.

        Args:
            temp_gguf_file: Fixture providing a mock GGUF file.
        """
        result = infer_model_name(str(temp_gguf_file))
        assert result == "test_model", f"Expected 'test_model', got '{result}'"

    def test_infer_model_name_nested_path(self, temp_output_dir: Path) -> None:
        """Test extracting name from nested directory path.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        nested_dir = temp_output_dir / "models" / "nested" / "My-Model-v2"
        nested_dir.mkdir(parents=True)
        (nested_dir / "config.json").write_text("{}")

        result = infer_model_name(str(nested_dir))
        assert result == "My-Model-v2", f"Expected 'My-Model-v2', got '{result}'"


# =============================================================================
# CONVERTER DOWNLOAD TESTS
# =============================================================================


class TestConverterDownload:
    """Tests for converter download functionality."""

    def test_download_converter(self, temp_output_dir: Path) -> None:
        """Test downloading the HF-to-GGUF converter script.

        Verifies that the converter script can be downloaded from the
        official llama.cpp repository.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        converter_path = temp_output_dir / "convert_hf_to_gguf.py"

        # Download the converter
        download_converter(converter_path)

        # Verify it was downloaded and has content
        assert converter_path.exists(), "Converter file should exist after download"
        content = converter_path.read_text()
        assert len(content) > 0, "Converter file should not be empty"
        assert "convert" in content.lower(), "Converter should contain conversion-related content"

    def test_download_converter_idempotent(self, temp_output_dir: Path) -> None:
        """Test that downloading twice overwrites the file.

        Args:
            temp_output_dir: Fixture providing a temporary directory.
        """
        converter_path = temp_output_dir / "convert_hf_to_gguf.py"

        # Download twice
        download_converter(converter_path)
        first_size = converter_path.stat().st_size

        download_converter(converter_path)
        second_size = converter_path.stat().st_size

        assert first_size == second_size, "File sizes should match after re-download"


# =============================================================================
# MODEL CONVERSION TESTS
# =============================================================================


class TestModelConversion:
    """Tests for HuggingFace to GGUF conversion."""

    @pytest.mark.slow
    def test_convert_hf_to_gguf(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test converting a HuggingFace model to GGUF format.

        This is a slow test as it involves downloading and running
        the converter script.

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        output_gguf = temp_output_dir / "converted_model.gguf"

        # Perform the conversion
        convert_hf_to_gguf(str(test_model_path), str(output_gguf))

        # Verify the output file was created
        assert output_gguf.exists(), f"Converted GGUF file should exist at {output_gguf}"
        assert output_gguf.stat().st_size > 0, "Converted GGUF file should not be empty"

    def test_convert_hf_to_gguf_invalid_input(self, temp_output_dir: Path) -> None:
        """Test conversion fails with invalid input path.

        Args:
            temp_output_dir: Fixture providing a temporary output directory.
        """
        invalid_path = temp_output_dir / "not_a_real_model"
        output_gguf = temp_output_dir / "output.gguf"

        with pytest.raises(RuntimeError):
            convert_hf_to_gguf(str(invalid_path), str(output_gguf))


# =============================================================================
# QUANTIZATION TESTS (SLOW)
# =============================================================================


@pytest.mark.slow
class TestQuantization:
    """Tests for model quantization using llama-quantize.

    These tests are marked as slow because they involve running
    the llama-quantize binary on actual model files.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_binaries(self, request: "FixtureRequest") -> None:
        """Skip tests if required binaries are not available."""
        binaries_available = request.getfixturevalue("binaries_available")
        if not binaries_available:
            pytest.skip("Required binaries not available")

    @pytest.mark.slow
    def test_quantize_model_q4_k_m(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test quantization using Q4_K_M type.

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        # First convert the model
        base_gguf = temp_output_dir / "base.gguf"
        convert_hf_to_gguf(str(test_model_path), str(base_gguf))

        # Now quantize it
        quant_gguf = temp_output_dir / "quantized_q4_k_m.gguf"
        quant_type = QuantizationType(id=15, name="Q4_K_M")

        quantize_model(str(base_gguf), str(quant_gguf), quant_type)

        # Verify output
        assert quant_gguf.exists(), "Quantized file should exist"
        assert quant_gguf.stat().st_size > 0, "Quantized file should not be empty"
        # Quantized file should be smaller than base
        assert quant_gguf.stat().st_size < base_gguf.stat().st_size, (
            "Quantized file should be smaller than base"
        )

    @pytest.mark.slow
    def test_quantize_model_multiple_types(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test quantizing to multiple quantization types.

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        # First convert the model
        base_gguf = temp_output_dir / "base.gguf"
        convert_hf_to_gguf(str(test_model_path), str(base_gguf))

        # Test multiple quantization types
        quant_types = [
            QuantizationType(id=2, name="Q4_0"),
            QuantizationType(id=3, name="Q4_1"),
            QuantizationType(id=15, name="Q4_K_M"),
        ]

        for quant_type in quant_types:
            quant_gguf = temp_output_dir / f"quantized_{quant_type.name}.gguf"
            quantize_model(str(base_gguf), str(quant_gguf), quant_type)

            assert quant_gguf.exists(), f"Quantized file for {quant_type.name} should exist"
            assert quant_gguf.stat().st_size > 0, (
                f"Quantized file for {quant_type.name} should not be empty"
            )


# =============================================================================
# BENCHMARK TESTS (SLOW)
# =============================================================================


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


# =============================================================================
# OUTPUT PARSING TESTS
# =============================================================================


class TestOutputParsing:
    """Tests for parsing llama-bench output."""

    def test_parse_llama_bench_output_valid(self) -> None:
        """Test parsing valid llama-bench markdown output."""
        sample_output = """
| model | size | params | backend | ngl | test | t/s |
| ------|------|--------|---------|-----|------|-----|
| test_model | 4.70 GiB | 4.02 B | CPU | 0 | pp512 | 1885.74 ± 2.76 |
| test_model | 4.70 GiB | 4.02 B | CPU | 0 | tg128 | 45.23 ± 0.12 |
"""
        results = parse_llama_bench_output(sample_output)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Check first result
        assert results[0]["model"] == "test_model"
        assert results[0]["test"] == "pp512"
        assert results[0]["tokens_per_sec"] == 1885.74
        assert results[0]["std_dev"] == 2.76
        assert results[0]["size_gib"] == 4.70

        # Check second result
        assert results[1]["test"] == "tg128"
        assert results[1]["tokens_per_sec"] == 45.23

    def test_parse_llama_bench_output_empty(self) -> None:
        """Test parsing empty output returns empty list."""
        results = parse_llama_bench_output("")
        assert results == [], "Empty output should return empty list"

    def test_parse_llama_bench_output_no_data(self) -> None:
        """Test parsing output with no data rows."""
        sample_output = """
| model | size | params | backend | ngl | test | t/s |
| ------|------|--------|---------|-----|------|-----|
"""
        results = parse_llama_bench_output(sample_output)
        assert results == [], "Output with no data should return empty list"

    def test_parse_llama_bench_output_malformed(self) -> None:
        """Test parsing malformed output handles errors gracefully."""
        sample_output = """
| model | size | params | backend | ngl | test | t/s |
| bad line without proper format
| test_model | invalid | data | CPU | 0 | pp512 | not_a_number |
"""
        results = parse_llama_bench_output(sample_output)
        # Should handle gracefully and return empty or partial results
        assert isinstance(results, list), "Should return a list even for bad input"


# =============================================================================
# END-TO-END INTEGRATION TEST
# =============================================================================


@pytest.mark.slow
class TestFullWorkflow:
    """End-to-end integration tests covering the complete workflow.

    These tests verify the full pipeline from HF model to benchmark results.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_binaries(self, request: "FixtureRequest") -> None:
        """Skip tests if required binaries are not available."""
        binaries_available = request.getfixturevalue("binaries_available")
        if not binaries_available:
            pytest.skip("Required binaries not available")

    @pytest.mark.slow
    def test_full_workflow_hf_to_benchmark(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Complete workflow: HF model -> GGUF -> Quantize -> Benchmark.

        This test verifies the entire pipeline works correctly:
        1. Detect HF model
        2. Convert to GGUF
        3. Quantize the model
        4. Run benchmarks

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        # Step 1: Verify HF model detection
        assert is_huggingface_model(str(test_model_path)), (
            "Test model should be detected as HF model"
        )

        # Step 2: Convert to GGUF
        model_name = infer_model_name(str(test_model_path))
        base_gguf = temp_output_dir / f"{model_name}-base.gguf"
        convert_hf_to_gguf(str(test_model_path), str(base_gguf))

        assert base_gguf.exists(), "Base GGUF should be created"
        assert is_gguf_file(str(base_gguf)), "Should be detected as GGUF file"

        # Step 3: Quantize to a single type
        quant_gguf = temp_output_dir / f"{model_name}-Q4_K_M.gguf"
        quant_type = QuantizationType(id=15, name="Q4_K_M")
        quantize_model(str(base_gguf), str(quant_gguf), quant_type)

        assert quant_gguf.exists(), "Quantized GGUF should be created"

        # Step 4: Run benchmarks
        results = run_full_benchmark(str(quant_gguf), quant_type.name, extra_args=[])

        # Verify we got results
        assert len(results) > 0, "Should have benchmark results"

        # Verify result structure
        for result in results:
            assert result.quant_type == "Q4_K_M"
            assert result.model_size_gib > 0
            assert result.tokens_per_sec > 0

    @pytest.mark.slow
    def test_full_workflow_with_quantization_dir(
        self,
        test_model_path: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with custom quantization directory.

        Args:
            test_model_path: Fixture providing the path to test HF model.
            temp_output_dir: Fixture providing a temporary output directory.
        """
        # Create a subdirectory for quants
        quant_dir = temp_output_dir / "quantizations"
        quant_dir.mkdir()

        # Convert and save to quant dir
        model_name = infer_model_name(str(test_model_path))
        base_gguf = quant_dir / f"{model_name}-base.gguf"
        convert_hf_to_gguf(str(test_model_path), str(base_gguf))

        # Quantize
        quant_gguf = quant_dir / f"{model_name}-Q4_0.gguf"
        quant_type = QuantizationType(id=2, name="Q4_0")
        quantize_model(str(base_gguf), str(quant_gguf), quant_type)

        # Verify files are in the right place
        assert base_gguf.exists(), "Base GGUF should be in quant dir"
        assert quant_gguf.exists(), "Quantized GGUF should be in quant dir"
