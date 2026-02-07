# llama-quant-benchmark

This repository contains scripts for testing model quantizations via llama-bench.

## Requirements

- `python`, preferably 3.14 but the script should work with older versions too.
- `llama-bench` and `llama-quantize` from `llama.cpp` project - should be available system-wide, for local installs add the directory to PATH before running the script.
- Raw model weights to be quantized and tested, either in HuggingFace format or GGUF - i've tested the script with Qwen3-4B.

If the script is pointed to a model in HuggingFace format, that model will be first converted to GGUF via [`convert_hf_to_gguf.py`](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py) script from [`llama.cpp`](https://github.com/ggml-org/llama.cpp) repository. That script will be automatically downloaded, no additional action by the user is necessary.

It's strongly recommended to create a virtualenv for running the script:

- `python -m venv .venv` for global Python installs
- `uv venv` when using `uv`

Afterwards, activate the virtualenv by sourcing the activation script in `.venv/bin` (dependent on the shell you're using)

## Usage

`python llama-quant-bench.py`

### Required arguments

- `--model <path to directory with raw model weights from huggingface, or GGUF file>`

Model name will be inferred from directory name (in case of HuggingFace model format) or GGUF name.

### Optional arguments

- `--quant-dir <path to directory where generated quants will be placed>` - by default, script creates a temporary directory (via `tempdir` module) for all the quants.
- `--model-name <model name>` - this can be used to override the model name, instead of using the name of directory with weights or GGUF file name.
- `--keep-quants` - by default, script will delete generated quants after performing benchmarks. Passing this argument disables that and lets you keep generated quants.
- `--quants <list of quantization types to benchmark, separated by commas>` - by default, script will test all the available quants. This argument can be used to test only a subset.
- `--output <path to output file>` - by default, the output will be placed in `quant-benchmark-report.md` file in current working directory.
- `--group <quant|test>` - group results by quantization type or test type (default: `quant`). When grouped, horizontal separators are added between groups in the table.

Remaining arguments will be passed to `llama-bench` for each benchmark (except `--model`, `-h`, `--help`, `--list-devices` which will be filtered out).
For a list of available quantizations to put in `--quants`, run `llama-quantize --help`.
For `--quants` argument, you can use either the names (e.g. `Q5_K,Q4_K,Q3_K_S` - letter case does not matter) or IDs (e.g. `17,15,11`), but they should not be mixed (e.g. `17,Q4_K` will throw an error).

## Generated report

The script will report progress (and the output from used tools) on standard output, and after performing all the benchmarks it will produce a comprehensive report in Markdown format.
The following shows that report's structure:

```md
# llama-quant-benchmark for `<model name>` (`<amount of model parameters>`)

| Quantization | Model size |  Test | Tokens/second |
|--------------|------------|-------|---------------|
|     QX_A     |  S.SS GiB  | pp512 | AAAA +/- B.BB |
|     QX_A     |  S.SS GiB  | tg128 | CCCC +/- D.DD |
|--------------|------------|-------|---------------|
|     QY_B     |  S.SS GiB  | pp512 | EEEE +/- F.FF |
|     QY_B     |  S.SS GiB  | tg128 | GGGG +/- H.HH |

Generated on `<current date/time>`
Used backend: `<backend used by llama-bench>`
Additional `llama-bench` arguments: `<user-provided arguments to llama-bench>`
```
