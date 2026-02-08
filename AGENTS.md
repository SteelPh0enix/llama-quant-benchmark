# Agents guidelines for llama-quant-benchmark

- Use `uv` for project management
  - There should be a virtual environment in `.venv/` subdirectory. If it's missing, create it. Use Python 3.13 and install the dependencies with `uv sync --all-extras`.
  - Always lint and format the code after making changes - use `uv tool run <tool_name>`, look for available tools in pyproject.toml
  - If there are any linting/formatting issues, first try to automatically fix them using available tools, and if they cannot be fixed that way - do it manually.
