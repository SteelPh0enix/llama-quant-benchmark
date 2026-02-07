"""Tests for model quantization using llama-quantize."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llama_quant_bench import (
    QuantizationType,
    convert_hf_to_gguf,
    quantize_model,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


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
