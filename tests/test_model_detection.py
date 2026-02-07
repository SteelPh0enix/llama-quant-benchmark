"""Tests for model detection functionality."""

from pathlib import Path

from llama_quant_bench import is_gguf_file, is_huggingface_model


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
