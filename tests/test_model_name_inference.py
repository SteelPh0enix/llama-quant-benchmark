"""Tests for model name inference functionality."""

from pathlib import Path

from llama_quant_bench import infer_model_name


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
