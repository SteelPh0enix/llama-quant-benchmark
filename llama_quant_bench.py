#!/usr/bin/env python3
"""LLaMA Quantization Benchmark Tool.

Benchmarks different quantization types using llama-quantize and llama-bench.
"""

import argparse
import datetime
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Any

# =============================================================================
# EXIT CODES
# =============================================================================


class ExitCode(IntEnum):
    """Exit codes for the benchmarking script."""

    SUCCESS = 0
    INVALID_ARGUMENTS = 1
    INVALID_MODEL_PATH = 2
    QUANT_FETCH_FAILED = 3
    QUANT_PARSE_ERROR = 4
    MODEL_CONVERSION_FAILED = 5
    NO_RESULTS = 6


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

DEFAULT_PERPLEXITY_TESTS: tuple[int, ...] = (512, 1024, 2048)  # pp512, pp1024, pp2048
DEFAULT_TOKEN_GENERATION_TESTS: tuple[int, ...] = (128, 256, 512)  # tg128, tg256, tg512
DEFAULT_GROUPING = "quant"  # Default grouping: "quant" or "test"

CONVERTER_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py"
)

# Regex pattern for parsing quantization types from llama-quantize --help
# Matches lines like "15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B"
QUANT_PATTERN = re.compile(r"^\s*(\d+)\s+or\s+(\S+)\s*:", re.MULTILINE)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BenchmarkResult:
    """Represents a single benchmark result."""

    quant_type: str
    model_size_gib: float
    model_params: str
    test_name: str
    tokens_per_sec: float
    std_dev: float


@dataclass
class BenchmarkReport:
    """Represents the complete benchmark report."""

    model_name: str
    model_params: str
    backend: str
    llama_bench_args: list[str]
    results: list[BenchmarkResult]
    generated_at: datetime.datetime


@dataclass
class QuantizationType:
    """Represents a quantization type."""

    id: int
    name: str


# Error messages for TRY003 compliance
ERR_NO_QUANTS = "No quantization types specified"
ERR_MIXED_NAMES_IDS = (
    "Cannot mix quantization names and IDs. "
    "Use either names (e.g., Q4_K) or IDs (e.g., 15) but not both."
)
ERR_UNKNOWN_QUANT = "Unknown quantization type: {}"
ERR_QUANT_FAILED = "Quantization failed: {}"
ERR_BENCH_FAILED = "Benchmark failed: {}"
ERR_QUANTIZE_NOT_FOUND = "llama-quantize not found in PATH"
ERR_INVALID_TEST_VALUES = "Invalid test values: {}"
ERR_EMPTY_TEST_VALUES = "Empty test values"
ERR_NEGATIVE_TEST_VALUE = "Test values must be positive integers, got: {}"
ERR_TEST_VALUE_TOO_LARGE = "Test value {} exceeds maximum allowed value of {}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def run_subprocess(cmd: list[str], error_msg: str) -> str:
    """Run a subprocess command and return combined output.

    Args:
        cmd: Command and arguments to run
        error_msg: Error message template for failures

    Returns:
        Combined stdout and stderr output

    Raises:
        RuntimeError: If the command fails
    """
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    output = result.stdout
    if result.stderr:
        output += result.stderr
        print(result.stderr, file=sys.stderr)

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(error_msg.format(result.stderr or result.stdout))

    return output


MAX_TEST_VALUE = 100000  # Maximum allowed test value to prevent memory issues


def parse_test_values(value: str) -> list[int]:
    """Parse comma-separated test values from CLI argument.

    Args:
        value: Comma-separated string of integers

    Returns:
        List of parsed positive integers

    Raises:
        ValueError: If value is empty, contains non-positive integers,
            or values exceed maximum allowed
    """
    if not value.strip():
        raise ValueError(ERR_EMPTY_TEST_VALUES)

    values = [int(x.strip()) for x in value.split(",")]

    for v in values:
        if v <= 0:
            raise ValueError(ERR_NEGATIVE_TEST_VALUE.format(v))
        if v > MAX_TEST_VALUE:
            raise ValueError(ERR_TEST_VALUE_TOO_LARGE.format(v, MAX_TEST_VALUE))

    return values


# =============================================================================
# QUANTIZATION TYPE MANAGEMENT
# =============================================================================


def get_available_quants() -> dict[str, QuantizationType]:
    """Get available quantization types by running llama-quantize --help.

    Returns a mapping of names/IDs to QuantizationType objects.
    """
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        raise RuntimeError(ERR_QUANTIZE_NOT_FOUND)

    output = run_subprocess([quantize_bin, "--help"], "Failed to get quantization types: {}")

    # Parse the "Allowed quantization types:" section
    quants: dict[str, QuantizationType] = {}

    for match in QUANT_PATTERN.finditer(output):
        qid = int(match.group(1))
        name = match.group(2)
        qt = QuantizationType(id=qid, name=name)
        quants[name.upper()] = qt
        quants[str(qid)] = qt

    return quants


def parse_user_quants(
    user_input: str,
    available_quants: dict[str, QuantizationType],
) -> list[QuantizationType]:
    """Parse user-provided quantization list.

    Validates that all items are either names or IDs (not mixed).
    """
    items = [item.strip() for item in user_input.split(",")]

    if not items:
        raise ValueError(ERR_NO_QUANTS)

    # Check if mixing names and IDs
    has_name = any(not item.isdigit() for item in items)
    has_id = any(item.isdigit() for item in items)

    if has_name and has_id:
        raise ValueError(ERR_MIXED_NAMES_IDS)

    # Validate and return
    result: list[QuantizationType] = []
    seen_ids: set[int] = set()

    for item in items:
        lookup_key = item if item.isdigit() else item.upper()
        if lookup_key not in available_quants:
            raise ValueError(ERR_UNKNOWN_QUANT.format(item))
        qt = available_quants[lookup_key]
        # Avoid duplicates
        if qt.id not in seen_ids:
            seen_ids.add(qt.id)
            result.append(qt)

    return result


def get_default_quants(
    available_quants: dict[str, QuantizationType],
) -> list[QuantizationType]:
    """Get default list of all quantization types (unique by ID)."""
    seen_ids: set[int] = set()
    result: list[QuantizationType] = []

    for key, qt in available_quants.items():
        if key.isdigit() and qt.id not in seen_ids:
            seen_ids.add(qt.id)
            result.append(qt)

    # Sort by ID
    result.sort(key=lambda x: x.id)
    return result


# =============================================================================
# MODEL DETECTION AND CONVERSION
# =============================================================================


def is_huggingface_model(path: Path) -> bool:
    """Check if path points to a HuggingFace model directory."""
    if not path.is_dir():
        return False

    # Check for typical HF files
    hf_files = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
    ]
    return any((path / f).exists() for f in hf_files)


def is_gguf_file(path: Path) -> bool:
    """Check if path points to a GGUF file."""
    return path.is_file() and path.suffix == ".gguf"


def download_converter(output_path: Path) -> None:
    """Download the convert_hf_to_gguf.py script."""
    print(f"Downloading converter to {output_path}...")

    # On NixOS, try to use the NixOS CA certificate bundle
    nixos_cert_path = "/etc/ssl/certs/ca-certificates.crt"
    if Path(nixos_cert_path).exists():
        ssl_context = ssl.create_default_context(cafile=nixos_cert_path)
    else:
        ssl_context = ssl.create_default_context()

    with (
        urllib.request.urlopen(  # noqa: S310
            CONVERTER_URL, context=ssl_context, timeout=30
        ) as response,
        output_path.open("wb") as f,
    ):
        f.write(response.read())
    print("Converter downloaded successfully")


def convert_hf_to_gguf(hf_dir: Path, output_path: Path) -> None:
    """Convert HuggingFace model to GGUF format.

    Downloads the converter script to a temporary directory and cleans it up
    after conversion.
    """
    temp_converter_dir = tempfile.TemporaryDirectory()
    converter_path = Path(temp_converter_dir.name) / "convert_hf_to_gguf.py"

    try:
        download_converter(converter_path)

        print(f"Converting HuggingFace model from {hf_dir} to {output_path}...")

        cmd = [
            sys.executable,
            str(converter_path),
            str(hf_dir),
            "--outfile",
            str(output_path),
            "--outtype",
            "auto",
        ]

        run_subprocess(cmd, "Failed to convert model: {}")
        print("Conversion successful")
    finally:
        temp_converter_dir.cleanup()


def infer_model_name(model_path: Path) -> str:
    """Infer model name from path."""
    if model_path.is_dir():
        # HuggingFace directory - use directory name
        return model_path.name
    if model_path.is_file() and model_path.suffix == ".gguf":
        # GGUF file - use filename without extension
        return model_path.stem
    # Fallback
    return model_path.name


# =============================================================================
# LLAMA-BENCH OUTPUT PARSING
# =============================================================================


# Expected column indices for llama-bench output parsing
# Format: | model | size | params | backend | n_threads | test | t/s |
COLUMN_MODEL = 0
COLUMN_SIZE = 1
COLUMN_PARAMS = 2
COLUMN_BACKEND = 3
COLUMN_TEST = 5
COLUMN_TOKENS_PER_SEC = 6
MIN_COLUMNS_REQUIRED = 7


def parse_llama_bench_output(output: str) -> list[dict[str, Any]]:
    """Parse llama-bench markdown output.

    Returns a list of dictionaries with parsed data.
    """
    results: list[dict[str, Any]] = []

    for line in output.split("\n"):
        stripped = line.strip()
        # Look for lines that start with "|" and have data (not headers or separators)
        if stripped.startswith("|") and "±" in stripped:
            parts = [p.strip() for p in stripped.split("|") if p.strip()]

            if len(parts) >= MIN_COLUMNS_REQUIRED:
                try:
                    model = parts[COLUMN_MODEL]
                    size_str = parts[COLUMN_SIZE]
                    params_str = parts[COLUMN_PARAMS]
                    backend = parts[COLUMN_BACKEND]
                    test = parts[COLUMN_TEST]
                    t_s_str = parts[COLUMN_TOKENS_PER_SEC]

                    # Parse size (extract number from "4.70 GiB")
                    size_match = re.search(r"([\d.]+)\s*GiB", size_str)
                    size_gib = float(size_match.group(1)) if size_match else 0.0

                    # Parse t/s (split "1885.74 ± 2.76")
                    t_s_parts = t_s_str.split("±")
                    tokens_per_sec = float(t_s_parts[0].strip())
                    std_dev = float(t_s_parts[1].strip()) if len(t_s_parts) > 1 else 0.0

                    results.append(
                        {
                            "model": model,
                            "size_gib": size_gib,
                            "params": params_str,
                            "backend": backend,
                            "test": test,
                            "tokens_per_sec": tokens_per_sec,
                            "std_dev": std_dev,
                        }
                    )
                except (IndexError, ValueError):
                    # Skip malformed lines
                    continue

    return results


# =============================================================================
# BENCHMARKING
# =============================================================================


def quantize_model(input_path: Path, output_path: Path, quant_type: QuantizationType) -> None:
    """Quantize a model using llama-quantize."""
    print(f"Quantizing to {quant_type.name} (ID: {quant_type.id})...")
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        raise RuntimeError(ERR_QUANTIZE_NOT_FOUND)

    cmd = [quantize_bin, str(input_path), str(output_path), str(quant_type.id)]
    run_subprocess(cmd, ERR_QUANT_FAILED)

    print(f"Quantization successful: {output_path}")


def _filter_llama_bench_args(args: list[str]) -> list[str]:
    """Filter out llama-bench specific arguments that should not be passed through."""
    excluded = {
        "--model",
        "-m",
        "-h",
        "--help",
        "--list-devices",
        "-p",
        "-n",
        "-pg",
    }
    return [a for a in args if a not in excluded]


def run_benchmark_all_tests(
    model_path: Path,
    perplexity_tests: list[int],
    token_generation_tests: list[int],
    extra_args: list[str] | None = None,
) -> str:
    """Run llama-bench with all configured tests in a single session."""
    # Build comma-separated test values
    pp_values = ",".join(str(pp) for pp in perplexity_tests)
    tg_values = ",".join(str(tg) for tg in token_generation_tests)

    cmd = ["llama-bench", "-m", str(model_path), "-p", pp_values, "-n", tg_values]

    if extra_args:
        cmd.extend(_filter_llama_bench_args(extra_args))

    print(f"Running: {' '.join(cmd)}")
    return run_subprocess(cmd, ERR_BENCH_FAILED)


def run_full_benchmark(
    model_path: Path,
    quant_type: str,
    perplexity_tests: list[int],
    token_generation_tests: list[int],
    extra_args: list[str] | None = None,
) -> tuple[list[BenchmarkResult], str, str]:
    """Run all configured benchmarks for a model and return results.

    Returns a tuple of (results, backend, model_params).
    """
    results: list[BenchmarkResult] = []

    # Run single benchmark session with all tests
    output = run_benchmark_all_tests(
        model_path, perplexity_tests, token_generation_tests, extra_args
    )
    parsed = parse_llama_bench_output(output)

    # Create set of valid test names for filtering
    valid_tests = {f"pp{pp}" for pp in perplexity_tests} | {
        f"tg{tg}" for tg in token_generation_tests
    }

    backend = "unknown"
    model_params = "unknown"

    for p in parsed:
        # Extract backend and model_params from first result
        if backend == "unknown":
            backend = p["backend"]
        if model_params == "unknown":
            model_params = p["params"]

        # Only include results for tests we configured
        if p["test"] in valid_tests:
            results.append(
                BenchmarkResult(
                    quant_type=quant_type,
                    model_size_gib=p["size_gib"],
                    model_params=p["params"],
                    test_name=p["test"],
                    tokens_per_sec=p["tokens_per_sec"],
                    std_dev=p["std_dev"],
                ),
            )

    return results, backend, model_params


# =============================================================================
# REPORT GENERATION
# =============================================================================


def _format_row_quant(r: BenchmarkResult) -> str:
    """Format row for quant grouping."""
    size_str = f"{r.model_size_gib:.2f} GiB"
    tps_str = f"{r.tokens_per_sec:.2f} ± {r.std_dev:.2f}"
    return f"| {r.quant_type} | {size_str} | {r.test_name} | {tps_str} |"


def _format_row_test(r: BenchmarkResult) -> str:
    """Format row for test grouping."""
    size_str = f"{r.model_size_gib:.2f} GiB"
    tps_str = f"{r.tokens_per_sec:.2f} ± {r.std_dev:.2f}"
    return f"| {r.test_name} | {r.quant_type} | {size_str} | {tps_str} |"


def _generate_grouped_rows(
    results: list[BenchmarkResult],
    grouping: str,
) -> tuple[list[str], list[str]]:
    """Generate table rows grouped by quant type or test name.

    Returns tuple of (header_lines, data_lines).
    """
    if grouping == "quant":
        header = ["| Quantization | Model size | Test | Tokens/second |"]
        separator = ["| ------------ | ---------- | ---- | ------------- |"]
        sort_key = attrgetter("quant_type")
        group_key = attrgetter("quant_type")
        format_row = _format_row_quant
    else:  # grouping == "test"
        header = ["| Test | Quantization | Model size | Tokens/second |"]
        separator = ["| ---- | ------------ | ---------- | ------------- |"]
        sort_key = attrgetter("test_name")
        group_key = attrgetter("test_name")
        format_row = _format_row_test

    lines: list[str] = []
    sorted_results = sorted(results, key=sort_key)

    first_group = True
    for _group_val, group in groupby(sorted_results, key=group_key):
        if not first_group:
            lines.extend(separator)
        first_group = False

        for result in group:
            lines.append(format_row(result))

    return header + separator, lines


def generate_markdown_report(report: BenchmarkReport, grouping: str) -> str:
    """Generate markdown report with specified grouping."""
    lines: list[str] = []

    # Header
    lines.append(f"# llama-quant-benchmark for `{report.model_name}` ({report.model_params})")
    lines.append("")

    header_lines, data_lines = _generate_grouped_rows(report.results, grouping)
    lines.extend(header_lines)
    lines.extend(data_lines)

    # Footer
    lines.append("")
    lines.append(f"Generated on `{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"Used backend: `{report.backend}`")

    if report.llama_bench_args:
        lines.append(f"Additional `llama-bench` arguments: `{' '.join(report.llama_bench_args)}`")

    return "\n".join(lines)


def save_report(report: BenchmarkReport, output_path: Path, grouping: str) -> None:
    """Save the report to a file."""
    markdown = generate_markdown_report(report, grouping)
    output_path.write_text(markdown)
    print(f"\nReport saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark LLaMA model quantizations using llama-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llama-quant-bench.py --model /path/to/model
  python llama-quant-bench.py --model /path/to/model.gguf --quants Q4_K,Q5_K
  python llama-quant-bench.py --model /path/to/model --keep-quants --output my-report.md
  python llama-quant-bench.py --model /path/to/model --group test -t 8
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to directory with raw model weights from HuggingFace, or GGUF file",
    )
    parser.add_argument(
        "--quant-dir",
        help=(
            "Path to directory where generated quants will be placed (default: temporary directory)"
        ),
    )
    parser.add_argument(
        "--model-name",
        help="Override the model name (default: inferred from directory/GGUF name)",
    )
    parser.add_argument(
        "--keep-quants",
        action="store_true",
        help=(
            "Keep generated quants after benchmarking "
            "(only useful with --quant-dir, otherwise ignored)"
        ),
    )
    parser.add_argument(
        "--keep-original-gguf",
        action="store_true",
        help=(
            "Keep the original GGUF file after converting from HuggingFace "
            "(only useful with --quant-dir, otherwise ignored)"
        ),
    )
    parser.add_argument(
        "--quants",
        help="Comma-separated list of quantization types to benchmark (default: all available)",
    )
    parser.add_argument(
        "--output",
        default="quant-benchmark-report.md",
        help="Path to output file (default: quant-benchmark-report.md)",
    )
    parser.add_argument(
        "--group",
        choices=["quant", "test"],
        default=DEFAULT_GROUPING,
        help="Group results by quantization type or test type (default: quant)",
    )
    parser.add_argument(
        "--perplexity-tests",
        default=",".join(str(x) for x in DEFAULT_PERPLEXITY_TESTS),
        help="Comma-separated list of perplexity test values (default: 512,1024,2048)",
    )
    parser.add_argument(
        "--token-generation-tests",
        default=",".join(str(x) for x in DEFAULT_TOKEN_GENERATION_TESTS),
        help="Comma-separated list of token generation test values (default: 128,256,512)",
    )

    return parser


def validate_keep_flags(args: argparse.Namespace) -> None:
    """Validate that keep-* flags are only used with --quant-dir."""
    if not args.quant_dir:
        if args.keep_quants:
            print("Error: --keep-quants can only be used with --quant-dir")
            sys.exit(ExitCode.INVALID_ARGUMENTS)
        if args.keep_original_gguf:
            print("Error: --keep-original-gguf can only be used with --quant-dir")
            sys.exit(ExitCode.INVALID_ARGUMENTS)


def validate_model_path(model_path: Path) -> None:
    """Validate that the model path exists and is valid."""
    if not is_huggingface_model(model_path) and not is_gguf_file(model_path):
        msg = (
            f"Error: Model path '{model_path}' is neither a HuggingFace "
            "model directory nor a GGUF file"
        )
        print(msg)
        sys.exit(ExitCode.INVALID_MODEL_PATH)


def get_quantization_types(args: argparse.Namespace) -> list[QuantizationType]:
    """Fetch and parse quantization types."""
    print("Fetching available quantization types...")
    try:
        available_quants = get_available_quants()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Error fetching quantization types: {e}")
        sys.exit(ExitCode.QUANT_FETCH_FAILED)

    if not available_quants:
        print("Error: Could not parse quantization types from llama-quantize")
        sys.exit(ExitCode.QUANT_FETCH_FAILED)

    if args.quants:
        try:
            return parse_user_quants(args.quants, available_quants)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(ExitCode.QUANT_PARSE_ERROR)

    return get_default_quants(available_quants)


def setup_quant_dir(
    args: argparse.Namespace,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Setup quantization directory."""
    if args.quant_dir:
        quant_dir = Path(args.quant_dir)
        quant_dir.mkdir(parents=True, exist_ok=True)
        return quant_dir, None

    temp_dir = tempfile.TemporaryDirectory()
    return Path(temp_dir.name), temp_dir


def convert_model_if_needed(
    model_path: Path,
    quant_dir: Path,
    model_name: str,
) -> tuple[Path, Path | None]:
    """Convert HF model to GGUF if needed."""
    if is_huggingface_model(model_path):
        base_gguf_path = quant_dir / f"{model_name}-base.gguf"
        try:
            convert_hf_to_gguf(model_path, base_gguf_path)
            return base_gguf_path, base_gguf_path  # noqa: TRY300
        except (RuntimeError, OSError) as e:
            print(f"Error converting model: {e}")
            sys.exit(ExitCode.MODEL_CONVERSION_FAILED)

    return model_path, None


def benchmark_single_quantization(
    quant_type: QuantizationType,
    base_model_path: Path,
    quant_dir: Path,
    model_name: str,
    perplexity_tests: list[int],
    token_generation_tests: list[int],
    remaining_args: list[str],
    *,
    keep_quants: bool,
) -> tuple[list[BenchmarkResult], str, str]:
    """Process a single quantization type."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {quant_type.name}")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    backend = "unknown"
    model_params = "unknown"

    quant_path = quant_dir / f"{model_name}-{quant_type.name}.gguf"
    try:
        quantize_model(base_model_path, quant_path, quant_type)
    except (RuntimeError, OSError) as e:
        print(f"Error quantizing to {quant_type.name}: {e}")
        return results, backend, model_params

    try:
        bench_results, bench_backend, bench_params = run_full_benchmark(
            quant_path,
            quant_type.name,
            perplexity_tests,
            token_generation_tests,
            remaining_args,
        )
        results.extend(bench_results)
        # Update backend and model_params if not already set
        if backend == "unknown":
            backend = bench_backend
        if model_params == "unknown":
            model_params = bench_params

    except (RuntimeError, OSError) as e:
        print(f"Error benchmarking {quant_type.name}: {e}")

    if not keep_quants and quant_path.exists():
        quant_path.unlink()
        print(f"Deleted: {quant_path}")

    return results, backend, model_params


def run_all_benchmarks(
    quant_types: list[QuantizationType],
    base_model_path: Path,
    quant_dir: Path,
    model_name: str,
    perplexity_tests: list[int],
    token_generation_tests: list[int],
    remaining_args: list[str],
    *,
    keep_quants: bool,
) -> tuple[list[BenchmarkResult], str, str]:
    """Run benchmarks for all quantization types."""
    all_results: list[BenchmarkResult] = []
    backend = "unknown"
    model_params = "unknown"

    for quant_type in quant_types:
        results, be, mp = benchmark_single_quantization(
            quant_type,
            base_model_path,
            quant_dir,
            model_name,
            perplexity_tests,
            token_generation_tests,
            remaining_args,
            keep_quants=keep_quants,
        )
        all_results.extend(results)
        if be != "unknown" and backend == "unknown":
            backend = be
        if mp != "unknown" and model_params == "unknown":
            model_params = mp

    return all_results, backend, model_params


def cleanup_resources(
    temp_dir: tempfile.TemporaryDirectory[str] | None,
    base_gguf_path: Path | None,
    *,
    keep_original_gguf: bool,
) -> None:
    """Cleanup temporary resources."""
    if temp_dir:
        temp_dir.cleanup()
    elif base_gguf_path and not keep_original_gguf and base_gguf_path.exists():
        base_gguf_path.unlink()
        print(f"Deleted base GGUF: {base_gguf_path}")


def main() -> None:
    """Run the benchmarking tool."""
    parser = create_argument_parser()
    args, remaining_args = parser.parse_known_args()

    validate_keep_flags(args)
    model_path = Path(args.model)
    validate_model_path(model_path)

    model_name = args.model_name if args.model_name else infer_model_name(model_path)
    print(f"Model name: {model_name}")

    # Parse test values from CLI arguments
    try:
        perplexity_tests = parse_test_values(args.perplexity_tests)
    except ValueError as e:
        msg = ERR_INVALID_TEST_VALUES.format(e)
        print(msg)
        sys.exit(ExitCode.INVALID_ARGUMENTS)
    try:
        token_generation_tests = parse_test_values(args.token_generation_tests)
    except ValueError as e:
        msg = ERR_INVALID_TEST_VALUES.format(e)
        print(msg)
        sys.exit(ExitCode.INVALID_ARGUMENTS)

    print(f"Perplexity tests: {perplexity_tests}")
    print(f"Token generation tests: {token_generation_tests}")

    quant_types = get_quantization_types(args)
    print(f"Will benchmark {len(quant_types)} quantization types: {[t.name for t in quant_types]}")

    quant_dir, temp_dir = setup_quant_dir(args)
    print(f"Quantization directory: {quant_dir}")

    base_model_path, base_gguf_path = convert_model_if_needed(model_path, quant_dir, model_name)

    try:
        all_results, backend, model_params = run_all_benchmarks(
            quant_types,
            base_model_path,
            quant_dir,
            model_name,
            perplexity_tests,
            token_generation_tests,
            remaining_args,
            keep_quants=args.keep_quants,
        )
    finally:
        cleanup_resources(temp_dir, base_gguf_path, keep_original_gguf=args.keep_original_gguf)

    if not all_results:
        print("\nError: No benchmark results collected")
        sys.exit(ExitCode.NO_RESULTS)

    output_path = Path(args.output)
    report = BenchmarkReport(
        model_name=model_name,
        model_params=model_params,
        backend=backend,
        llama_bench_args=remaining_args,
        results=all_results,
        generated_at=datetime.datetime.now(tz=datetime.UTC),
    )

    save_report(report, output_path, args.group)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
