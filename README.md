# llama-quant-benchmark

This repository contains scripts for testing model quantizations via llama-bench.

## Requirements

- `python`, version 3.13 or higher
- `llama-bench` and `llama-quantize` from `llama.cpp` project - should be available system-wide, for local installs add the directory to PATH before running the script
- Raw model weights to be quantized and tested, either in HuggingFace format or GGUF - i've tested the script with Qwen3-4B

### System Requirements

- **Disk space**: Approximately 2-3x the size of the original model (for intermediate quantized files)
- **RAM**: Varies by model size (recommend 16GB+ for 7B models)
- **CPU**: Multi-core CPU recommended for faster benchmarking

If the script is pointed to a model in HuggingFace format, that model will be first converted to GGUF via [`convert_hf_to_gguf.py`](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py) script from [`llama.cpp`](https://github.com/ggml-org/llama.cpp) repository. That script will be automatically downloaded, no additional action by the user is necessary.

## Installation

### Prerequisites

Ensure you have Python 3.13+ and `llama-bench`/`llama-quantize` from llama.cpp installed and available in your PATH.

### Setup

It's strongly recommended to create a virtual environment for running the script:

**Using uv (recommended):**

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

**Using standard Python venv:**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

```bash
python llama_quant_bench.py --model <path>
```

### Required arguments

- `--model <path>` - Path to directory with raw model weights from HuggingFace, or path to a GGUF file

Model name will be inferred from directory name (in case of HuggingFace model format) or GGUF filename.

### Optional arguments

- `--quant-dir <path>` - Directory where generated quants will be placed (default: temporary directory)
- `--model-name <name>` - Override the model name (instead of using directory name or GGUF filename)
- `--keep-quants` - Keep generated quants after benchmarking (only useful with `--quant-dir`)
- `--keep-original-gguf` - Keep the original GGUF file after converting from HuggingFace (only useful with `--quant-dir`)
- `--quants <list>` - Comma-separated list of quantization types to benchmark (default: all available)
- `--output-dir <path>` - Directory to save reports (default: current directory)
- `--perplexity-tests <values>` - Comma-separated list of perplexity test values (default: `512,1024,2048`)
- `--token-generation-tests <values>` - Comma-separated list of token generation test values (default: `128,256,512`)

**Note:** The `--keep-quants` and `--keep-original-gguf` flags require `--quant-dir` to be set. When using a temporary directory (default behavior), all files are always deleted after benchmarking.

Remaining arguments will be passed to `llama-bench` for each benchmark (except `--model`, `-h`, `--help`, `--list-devices` which will be filtered out).

For a list of available quantizations to put in `--quants`, run `llama-quantize --help`.
For `--quants` argument, you can use either the names (e.g. `Q5_K,Q4_K,Q3_K_S` - letter case does not matter) or IDs (e.g. `17,15,11`), but they should not be mixed (e.g. `17,Q4_K` will throw an error).

### Examples

```bash
# Benchmark all quantizations for a HuggingFace model
python llama_quant_bench.py --model /path/to/hf-model

# Benchmark specific quantizations for a GGUF file
python llama_quant_bench.py --model /path/to/model.gguf --quants Q4_K,Q5_K

# Keep quantized files in a specific directory
python llama_quant_bench.py --model /path/to/model --quant-dir ./quants --keep-quants

# Custom perplexity and token generation tests
python llama_quant_bench.py --model /path/to/model --perplexity-tests 256,512 --token-generation-tests 64,128

# Pass additional arguments to llama-bench (e.g., use 8 threads)
python llama_quant_bench.py --model /path/to/model -t 8
```

## Generated reports

The script will report progress (and the output from used tools) on standard output, and after performing all the benchmarks it will produce two comprehensive reports in Markdown format, saved to the specified output directory (current directory by default).

Reports are timestamped to prevent overwriting previous results. The filenames follow this pattern:

- `<model_name>_by-quant_YYYYMMDD_HHMMSS.md` - Results grouped by quantization type
- `<model_name>_by-test_YYYYMMDD_HHMMSS.md` - Results grouped by test type

### Report grouped by quantization type

```markdown
# llama-quant-benchmark for `<model name>` (`<amount of model parameters>`)

| Quantization | Model size | Test  | Tokens/second |
|--------------|------------|-------|---------------|
|     QX_A     |  S.SS GiB  | pp512 | AAAA ± B.BB   |
|     QX_A     |  S.SS GiB  | tg128 | CCCC ± D.DD   |
|--------------|------------|-------|---------------|
|     QY_B     |  S.SS GiB  | pp512 | EEEE ± F.FF   |
|     QY_B     |  S.SS GiB  | tg128 | GGGG ± H.HH   |

Generated on `<current date/time>`
Used backend: `<backend used by llama-bench>`
Additional `llama-bench` arguments: `<user-provided arguments to llama-bench>`
```

### Report grouped by test type

```markdown
# llama-quant-benchmark for `<model name>` (`<amount of model parameters>`)

| Test  | Quantization | Model size | Tokens/second |
|-------|--------------|------------|---------------|
| pp512 |     QX_A     |  S.SS GiB  | AAAA ± B.BB   |
| pp512 |     QY_B     |  S.SS GiB  | EEEE ± F.FF   |
|-------|--------------|------------|---------------|
| tg128 |     QX_A     |  S.SS GiB  | CCCC ± D.DD   |
| tg128 |     QY_B     |  S.SS GiB  | GGGG ± H.HH   |

Generated on `<current date/time>`
Used backend: `<backend used by llama-bench>`
Additional `llama-bench` arguments: `<user-provided arguments to llama-bench>`
```

## Troubleshooting

### llama-quantize not found

Ensure that `llama-quantize` and `llama-bench` from the llama.cpp project are built and available in your PATH. You can verify this by running:

```bash
which llama-quantize
which llama-bench
```

### Conversion fails

If converting from HuggingFace format fails:

- Ensure the model directory contains valid HuggingFace format files (`config.json`, `model.safetensors` or `pytorch_model.bin`, etc.)
- Check that you have sufficient disk space (2-3x the model size)
- Verify your internet connection (the converter script is downloaded from GitHub)

### Out of memory errors

If you encounter OOM errors during benchmarking:

- Reduce perplexity test values: `--perplexity-tests 256,512`
- Reduce token generation test values: `--token-generation-tests 64,128`
- Use smaller quantization types

### Slow performance

- Use `--quant-dir` with an SSD for faster I/O
- Ensure your CPU supports the optimizations used by llama.cpp
- Pass thread count to llama-bench: `-t 8`

## License

This project is licensed under the AGPL v3 License - see the [LICENSE](LICENSE) file for details.
