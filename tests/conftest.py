"""Pytest configuration and fixtures for llama-quant-benchmark tests."""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

TEST_MODEL_ENV_VAR = "LLAMA_QUANT_BENCH_TEST_MODEL"

# =============================================================================
# SKIP CHECKS
# =============================================================================


def _is_binary_available(binary_name: str) -> bool:
    """Check if a binary is available in PATH."""
    return shutil.which(binary_name) is not None


# Check for required binaries
LLAMA_QUANTIZE_AVAILABLE = _is_binary_available("llama-quantize")
LLAMA_BENCH_AVAILABLE = _is_binary_available("llama-bench")

# Skip reasons
BINARIES_SKIP_REASON = "Required binaries (llama-quantize, llama-bench) not found in PATH"
MODEL_PATH_SKIP_REASON = f"Test model path not set in {TEST_MODEL_ENV_VAR} environment variable"

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def test_model_path() -> Path:
    """Fixture providing the path to the test HuggingFace model.

    Reads from LLAMA_QUANT_BENCH_TEST_MODEL environment variable.

    Returns:
        Path to the HuggingFace model directory.

    Raises:
        pytest.skip: If the model path is not set or invalid.
    """
    # Get path from environment variable
    model_path_str = os.environ.get(TEST_MODEL_ENV_VAR)
    if not model_path_str:
        pytest.skip(f"{MODEL_PATH_SKIP_REASON}: Environment variable not set")
    assert model_path_str is not None
    model_path = Path(model_path_str).resolve()

    # Validate the path exists
    if not model_path.exists():
        pytest.skip(f"{MODEL_PATH_SKIP_REASON}: Path does not exist: {model_path}")

    if not model_path.is_dir():
        pytest.skip(f"{MODEL_PATH_SKIP_REASON}: Path is not a directory: {model_path}")

    # Check for required HF model files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (model_path / f).exists()]

    if missing_files:
        pytest.skip(f"{MODEL_PATH_SKIP_REASON}: Missing required files: {missing_files}")

    return model_path


@pytest.fixture
def temp_output_dir() -> Generator[Path]:
    """Fixture providing a temporary directory for test outputs.

    Creates a temporary directory that is cleaned up after each test.

    Yields:
        Path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp(prefix="llama_quant_bench_test_")
    temp_path = Path(temp_dir)

    try:
        yield temp_path
    finally:
        # Cleanup: remove the temporary directory and all its contents
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_hf_model_dir(temp_output_dir: Path) -> Path:
    """Fixture creating a mock HuggingFace model directory structure.

    Creates a minimal HF model directory with required files.

    Args:
        temp_output_dir: The temporary output directory fixture.

    Returns:
        Path to the mock HF model directory.
    """
    hf_dir = temp_output_dir / "mock_hf_model"
    hf_dir.mkdir(parents=True)

    # Create minimal config.json
    config_content = '{"model_type": "test", "architectures": ["TestModel"]}'
    (hf_dir / "config.json").write_text(config_content)

    return hf_dir


@pytest.fixture
def temp_gguf_file(temp_output_dir: Path) -> Path:
    """Fixture creating a mock GGUF file.

    Creates an empty file with .gguf extension for testing.

    Args:
        temp_output_dir: The temporary output directory fixture.

    Returns:
        Path to the mock GGUF file.
    """
    gguf_path = temp_output_dir / "test_model.gguf"
    # Create an empty file (just for path/extension testing)
    gguf_path.touch()
    return gguf_path


@pytest.fixture(scope="session")
def binaries_available() -> bool:
    """Fixture checking if required binaries are available.

    Returns:
        True if both llama-quantize and llama-bench are available.
    """
    return LLAMA_QUANTIZE_AVAILABLE and LLAMA_BENCH_AVAILABLE


# =============================================================================
# SKIP DECORATORS
# =============================================================================


requires_binaries = pytest.mark.skipif(
    not (LLAMA_QUANTIZE_AVAILABLE and LLAMA_BENCH_AVAILABLE),
    reason=BINARIES_SKIP_REASON,
)


def _test_model_path_exists() -> bool:
    """Check if test model path exists and is valid."""
    model_path_str = os.environ.get(TEST_MODEL_ENV_VAR)
    if not model_path_str:
        return False
    model_path = Path(model_path_str)
    return model_path.exists() and model_path.is_dir() and (model_path / "config.json").exists()


requires_test_model = pytest.mark.skipif(
    not _test_model_path_exists(),
    reason=MODEL_PATH_SKIP_REASON,
)
