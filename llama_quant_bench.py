#!/usr/bin/env python3
"""LLaMA Quantization Benchmark Tool

Benchmarks different quantization types using llama-quantize and llama-bench.
"""

import argparse
import datetime
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

# =============================================================================
# EXIT CODES - Explicit values for clarity
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
# GLOBAL CONFIGURATION - Easy to modify
# =============================================================================

PERPLEXITY_TESTS = [512, 1024, 2048]  # pp512, pp1024, pp2048
TOKEN_GENERATION_TESTS = [128, 256, 512]  # tg128, tg256, tg512
DEFAULT_GROUPING = "quant"  # Default grouping: "quant" or "test"

CONVERTER_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py"
)


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


# =============================================================================
# QUANTIZATION TYPE MANAGEMENT
# =============================================================================


def get_available_quants() -> dict[str, QuantizationType]:
    """Get available quantization types by running llama-quantize --help.
    Returns a mapping of names/IDs to QuantizationType objects.
    """
    result = subprocess.run(["llama-quantize", "--help"], capture_output=True, text=True)

    output = result.stdout + result.stderr

    # Parse the "Allowed quantization types:" section
    quants = {}

    # Regex pattern: matches lines like "15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B"
    # or "  0  or  F32     : 26.00G              @ 7B"
    quant_pattern = re.compile(r"^\s*(\d+)\s+or\s+(\S+)\s*:", re.MULTILINE)

    for match in quant_pattern.finditer(output):
        qid = int(match.group(1))
        name = match.group(2)
        qt = QuantizationType(id=qid, name=name)
        quants[name] = qt
        quants[name.lower()] = qt
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
        raise ValueError("No quantization types specified")

    # Check if mixing names and IDs
    has_name = False
    has_id = False

    for item in items:
        if item.isdigit():
            has_id = True
        else:
            has_name = True

    if has_name and has_id:
        raise ValueError(
            "Cannot mix quantization names and IDs. Use either names (e.g., Q4_K) or IDs (e.g., 15) but not both.",
        )

    # Validate and return
    result = []
    for item in items:
        lookup_key = item if item.isdigit() else item.lower()
        if lookup_key not in available_quants:
            raise ValueError(f"Unknown quantization type: {item}")
        qt = available_quants[lookup_key]
        # Avoid duplicates
        if qt not in result:
            result.append(qt)

    return result


def get_default_quants(
    available_quants: dict[str, QuantizationType],
) -> list[QuantizationType]:
    """Get default list of all quantization types (unique by ID)."""
    seen_ids = set()
    result = []

    for key, qt in available_quants.items():
        if key.isdigit():  # Only process numeric keys to avoid duplicates
            if qt.id not in seen_ids:
                seen_ids.add(qt.id)
                result.append(qt)

    # Sort by ID
    result.sort(key=lambda x: x.id)
    return result


# =============================================================================
# MODEL DETECTION AND CONVERSION
# =============================================================================


def is_huggingface_model(path: str) -> bool:
    """Check if path points to a HuggingFace model directory."""
    p = Path(path)
    if not p.is_dir():
        return False

    # Check for typical HF files
    hf_files = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
    ]
    return any((p / f).exists() for f in hf_files)


def is_gguf_file(path: str) -> bool:
    """Check if path points to a GGUF file."""
    p = Path(path)
    return p.is_file() and p.suffix == ".gguf"


def download_converter(output_path: Path) -> None:
    """Download the convert_hf_to_gguf.py script."""
    print(f"Downloading converter to {output_path}...")
    import ssl

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(CONVERTER_URL, context=ssl_context) as response:
        with open(output_path, "wb") as f:
            f.write(response.read())
    print("Converter downloaded successfully")


def convert_hf_to_gguf(hf_dir: str, output_path: str) -> None:
    """Convert HuggingFace model to GGUF format."""
    converter_path = Path(tempfile.gettempdir()) / "convert_hf_to_gguf.py"

    if not converter_path.exists():
        download_converter(converter_path)

    print(f"Converting HuggingFace model from {hf_dir} to {output_path}...")
    result = subprocess.run(
        [sys.executable, str(converter_path), hf_dir, "--outfile", output_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Conversion stdout: {result.stdout}")
        print(f"Conversion stderr: {result.stderr}")
        raise RuntimeError(f"Failed to convert model: {result.stderr}")

    print("Conversion successful")


def infer_model_name(model_path: str) -> str:
    """Infer model name from path."""
    p = Path(model_path)

    if p.is_dir():
        # HuggingFace directory - use directory name
        return p.name
    if p.is_file() and p.suffix == ".gguf":
        # GGUF file - use filename without extension
        return p.stem
    # Fallback
    return p.name


# =============================================================================
# LLAMA-BENCH OUTPUT PARSING
# =============================================================================


def parse_llama_bench_output(output: str) -> list[dict[str, Any]]:
    """Parse llama-bench markdown output.
    Returns a list of dictionaries with parsed data.
    """
    results = []

    # Split into lines and look for table rows
    lines = output.split("\n")

    for line in lines:
        line = line.strip()
        # Look for lines that start with "|" and have data (not headers or separators)
        if line.startswith("|") and "±" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]

            if len(parts) >= 6:
                # Expected: model, size, params, backend, ngl, test, t/s
                # parts[0] = model
                # parts[1] = size (e.g., "4.70 GiB")
                # parts[2] = params (e.g., "4.02 B")
                # parts[3] = backend
                # parts[4] = ngl
                # parts[5] = test (e.g., "pp512" or "tg128")
                # parts[6] = t/s (e.g., "1885.74 ± 2.76")

                try:
                    model = parts[0]
                    size_str = parts[1]
                    params_str = parts[2]
                    backend = parts[3]
                    test = parts[5]
                    t_s_str = parts[6]

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
                        },
                    )
                except (IndexError, ValueError):
                    # Skip malformed lines
                    continue

    return results


# =============================================================================
# BENCHMARKING
# =============================================================================


def quantize_model(input_path: str, output_path: str, quant_type: QuantizationType) -> None:
    """Quantize a model using llama-quantize."""
    print(f"Quantizing to {quant_type.name} (ID: {quant_type.id})...")
    result = subprocess.run(
        ["llama-quantize", input_path, output_path, str(quant_type.id)],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed: {result.stderr}")

    print(f"Quantization successful: {output_path}")


def run_benchmark(
    model_path: str,
    test_prompt: int | None = None,
    test_gen: int | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Run llama-bench and return the output."""
    cmd = ["llama-bench", "-m", model_path]

    if test_prompt:
        cmd.extend(["-p", str(test_prompt)])
    elif test_gen:
        cmd.extend(["-n", str(test_gen)])

    if extra_args:
        # Filter out llama-bench specific args that we don't want to pass
        filtered_args = [
            a for a in extra_args if a not in ("--model", "-m", "-h", "--help", "--list-devices")
        ]
        cmd.extend(filtered_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")

    return result.stdout + result.stderr


def run_benchmark_all_tests(model_path: str, extra_args: list[str] | None = None) -> str:
    """Run llama-bench with all configured tests in a single session."""
    # Build comma-separated test values
    pp_values = ",".join(str(pp) for pp in PERPLEXITY_TESTS)
    tg_values = ",".join(str(tg) for tg in TOKEN_GENERATION_TESTS)

    cmd = ["llama-bench", "-m", model_path, "-p", pp_values, "-n", tg_values]

    if extra_args:
        # Filter out llama-bench specific args that we don't want to pass
        filtered_args = [
            a
            for a in extra_args
            if a not in ("--model", "-m", "-h", "--help", "--list-devices", "-p", "-n")
        ]
        cmd.extend(filtered_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")

    return result.stdout + result.stderr


def run_full_benchmark(
    model_path: str,
    quant_type: str,
    extra_args: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run all configured benchmarks for a model and return results."""
    results = []

    # Run single benchmark session with all tests
    output = run_benchmark_all_tests(model_path, extra_args)
    parsed = parse_llama_bench_output(output)

    # Create set of valid test names for filtering
    valid_tests = {f"pp{pp}" for pp in PERPLEXITY_TESTS} | {
        f"tg{tg}" for tg in TOKEN_GENERATION_TESTS
    }

    for p in parsed:
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

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_markdown_report(report: BenchmarkReport, grouping: str) -> str:
    """Generate markdown report with specified grouping."""
    lines = []

    # Header
    lines.append(f"# llama-quant-benchmark for `{report.model_name}` ({report.model_params})")
    lines.append("")

    if grouping == "quant":
        # Group by quantization type
        lines.append("| Quantization | Model size | Test | Tokens/second |")
        lines.append("| ------------ | ---------- | ---- | ------------- |")

        # Group results by quant type
        from itertools import groupby

        sorted_results = sorted(report.results, key=lambda x: x.quant_type)

        first_group = True
        for quant_type, group in groupby(sorted_results, key=lambda x: x.quant_type):
            if not first_group:
                # Add separator between groups
                lines.append("| ------------ | ---------- | ---- | ------------- |")
            first_group = False

            for result in group:
                size_str = f"{result.model_size_gib:.2f} GiB"
                tps_str = f"{result.tokens_per_sec:.2f} ± {result.std_dev:.2f}"
                lines.append(
                    f"| {result.quant_type} | {size_str} | {result.test_name} | {tps_str} |",
                )

    else:  # grouping == "test"
        # Group by test type
        lines.append("| Test | Quantization | Model size | Tokens/second |")
        lines.append("| ---- | ------------ | ---------- | ------------- |")

        # Group results by test name
        from itertools import groupby

        sorted_results = sorted(report.results, key=lambda x: x.test_name)

        first_group = True
        for test_name, group in groupby(sorted_results, key=lambda x: x.test_name):
            if not first_group:
                # Add separator between groups
                lines.append("| ---- | ------------ | ---------- | ------------- |")
            first_group = False

            for result in group:
                size_str = f"{result.model_size_gib:.2f} GiB"
                tps_str = f"{result.tokens_per_sec:.2f} ± {result.std_dev:.2f}"
                lines.append(
                    f"| {result.test_name} | {result.quant_type} | {size_str} | {tps_str} |",
                )

    # Footer
    lines.append("")
    lines.append(f"Generated on `{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"Used backend: `{report.backend}`")

    if report.llama_bench_args:
        lines.append(f"Additional `llama-bench` arguments: `{' '.join(report.llama_bench_args)}`")

    return "\n".join(lines)


def save_report(report: BenchmarkReport, output_path: str, grouping: str) -> None:
    """Save the report to a file."""
    markdown = generate_markdown_report(report, grouping)

    with open(output_path, "w") as f:
        f.write(markdown)

    print(f"\nReport saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
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

    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="Path to directory with raw model weights from HuggingFace, or GGUF file",
    )

    # Optional arguments
    parser.add_argument(
        "--quant-dir",
        help="Path to directory where generated quants will be placed (default: temporary directory)",
    )
    parser.add_argument(
        "--model-name",
        help="Override the model name (default: inferred from directory/GGUF name)",
    )
    parser.add_argument(
        "--keep-quants",
        action="store_true",
        help="Keep generated quants after benchmarking (only useful with --quant-dir, otherwise ignored)",
    )
    parser.add_argument(
        "--keep-original-gguf",
        action="store_true",
        help="Keep the original GGUF file after converting from HuggingFace (only useful with --quant-dir, otherwise ignored)",
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

    # Remaining arguments go to llama-bench
    args, remaining_args = parser.parse_known_args()

    # Validate that keep-* flags are only used with --quant-dir
    if not args.quant_dir:
        if args.keep_quants:
            print("Error: --keep-quants can only be used with --quant-dir")
            sys.exit(ExitCode.INVALID_ARGUMENTS)
        if args.keep_original_gguf:
            print("Error: --keep-original-gguf can only be used with --quant-dir")
            sys.exit(ExitCode.INVALID_ARGUMENTS)

    # Validate model path
    model_path = args.model
    if not is_huggingface_model(model_path) and not is_gguf_file(model_path):
        print(
            f"Error: Model path '{model_path}' is neither a HuggingFace model directory nor a GGUF file",
        )
        sys.exit(ExitCode.INVALID_MODEL_PATH)

    # Get model name
    model_name = args.model_name if args.model_name else infer_model_name(model_path)
    print(f"Model name: {model_name}")

    # Get available quantization types
    print("Fetching available quantization types...")
    try:
        available_quants = get_available_quants()
    except Exception as e:
        print(f"Error fetching quantization types: {e}")
        sys.exit(ExitCode.QUANT_FETCH_FAILED)

    if not available_quants:
        print("Error: Could not parse quantization types from llama-quantize")
        sys.exit(ExitCode.QUANT_FETCH_FAILED)

    # Parse user-specified quants or use defaults
    if args.quants:
        try:
            quant_types = parse_user_quants(args.quants, available_quants)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(ExitCode.QUANT_PARSE_ERROR)
    else:
        quant_types = get_default_quants(available_quants)

    print(f"Will benchmark {len(quant_types)} quantization types")

    # Setup quantization directory
    if args.quant_dir:
        quant_dir = Path(args.quant_dir)
        quant_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory()
        quant_dir = Path(temp_dir.name)

    print(f"Quantization directory: {quant_dir}")

    # Convert HF model to GGUF if needed
    base_gguf_path = None
    if is_huggingface_model(model_path):
        base_gguf_path = quant_dir / f"{model_name}-base.gguf"
        try:
            convert_hf_to_gguf(model_path, str(base_gguf_path))
            base_model_path = str(base_gguf_path)
        except Exception as e:
            print(f"Error converting model: {e}")
            sys.exit(ExitCode.MODEL_CONVERSION_FAILED)
    else:
        base_model_path = model_path

    # Run benchmarks
    all_results = []
    backend = "unknown"
    model_params = "unknown"

    try:
        for quant_type in quant_types:
            print(f"\n{'=' * 60}")
            print(f"Benchmarking: {quant_type.name}")
            print(f"{'=' * 60}")

            # Quantize
            quant_path = quant_dir / f"{model_name}-{quant_type.name}.gguf"
            try:
                quantize_model(base_model_path, str(quant_path), quant_type)
            except Exception as e:
                print(f"Error quantizing to {quant_type.name}: {e}")
                continue

            # Benchmark
            try:
                results = run_full_benchmark(str(quant_path), quant_type.name, remaining_args)
                all_results.extend(results)

                # Extract backend and params from first result
                if results and backend == "unknown":
                    # Run a quick test to get backend info
                    test_output = run_benchmark(
                        str(quant_path),
                        test_prompt=512,
                        extra_args=remaining_args,
                    )
                    parsed = parse_llama_bench_output(test_output)
                    if parsed:
                        backend = parsed[0]["backend"]
                        model_params = parsed[0]["params"]

            except Exception as e:
                print(f"Error benchmarking {quant_type.name}: {e}")
                continue

            # Delete quant if not keeping
            if not args.keep_quants and quant_path.exists():
                quant_path.unlink()
                print(f"Deleted: {quant_path}")

    finally:
        # Cleanup
        if temp_dir:
            temp_dir.cleanup()
        elif base_gguf_path is not None:
            if not args.keep_original_gguf and base_gguf_path.exists():
                base_gguf_path.unlink()
                print(f"Deleted base GGUF: {base_gguf_path}")

    if not all_results:
        print("\nError: No benchmark results collected")
        sys.exit(ExitCode.NO_RESULTS)

    # Generate report
    report = BenchmarkReport(
        model_name=model_name,
        model_params=model_params,
        backend=backend,
        llama_bench_args=remaining_args,
        results=all_results,
        generated_at=datetime.datetime.now(),
    )

    save_report(report, args.output, args.group)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
