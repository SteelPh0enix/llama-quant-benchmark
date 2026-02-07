"""Tests for converter download functionality."""

from pathlib import Path

from llama_quant_bench import download_converter


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
