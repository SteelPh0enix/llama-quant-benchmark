# Agents guidelines for llama-quant-benchmark

 - Use `uv` for project management
   - There should be a virtual environment in `.venv/` subdirectory. If it's missing, create it. Use Python 3.13 and install the dependencies with `uv sync --all-extras`.
   - Always lint and format the code after making changes - use `uv tool run <tool_name>`, look for available tools in pyproject.toml
   - If there are any linting/formatting issues, first try to automatically fix them using available tools, and if they cannot be fixed that way - do it manually.

 - When implementing new functionality, always make sure it's properly tested, including edge cases.

# Coverage report generation

This project uses pytest-cov to track test coverage.

## Running tests with coverage

Run tests with coverage report:

```bash
uv tool run pytest
```

The coverage report will be:
- Displayed in console with missing lines shown
- Generated as HTML report in `htmlcov/` directory

## Coverage threshold

The project enforces a minimum **80% coverage threshold** for both line and branch coverage. The test suite will fail if coverage is below this threshold.

## Viewing HTML report

Open `htmlcov/index.html` in a browser to explore coverage interactively.
