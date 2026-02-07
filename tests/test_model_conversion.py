"""Tests for HuggingFace to GGUF conversion."""

from pathlib import Path

import pytest

from llama_quant_bench import convert_hf_to_gguf


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
