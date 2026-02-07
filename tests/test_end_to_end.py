"""End-to-end integration tests covering the complete workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llama_quant_bench import (
    QuantizationType,
    convert_hf_to_gguf,
    infer_model_name,
    is_gguf_file,
    is_huggingface_model,
    quantize_model,
    run_full_benchmark,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


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
