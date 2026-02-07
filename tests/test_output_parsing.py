"""Tests for parsing llama-bench output."""

from llama_quant_bench import parse_llama_bench_output


class TestOutputParsing:
    """Tests for parsing llama-bench output."""

    def test_parse_llama_bench_output_valid(self) -> None:
        """Test parsing valid llama-bench markdown output."""
        sample_output = """
 | model | size | params | backend | ngl | test | t/s |
 | ------|------|--------|---------|-----|------|-----|
 | test_model | 4.70 GiB | 4.02 B | CPU | 0 | pp512 | 1885.74 ± 2.76 |
 | test_model | 4.70 GiB | 4.02 B | CPU | 0 | tg128 | 45.23 ± 0.12 |
 """
        results = parse_llama_bench_output(sample_output)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Check first result
        assert results[0]["model"] == "test_model"
        assert results[0]["test"] == "pp512"
        assert results[0]["tokens_per_sec"] == 1885.74
        assert results[0]["std_dev"] == 2.76
        assert results[0]["size_gib"] == 4.70

        # Check second result
        assert results[1]["test"] == "tg128"
        assert results[1]["tokens_per_sec"] == 45.23

    def test_parse_llama_bench_output_empty(self) -> None:
        """Test parsing empty output returns empty list."""
        results = parse_llama_bench_output("")
        assert results == [], "Empty output should return empty list"

    def test_parse_llama_bench_output_no_data(self) -> None:
        """Test parsing output with no data rows."""
        sample_output = """
 | model | size | params | backend | ngl | test | t/s |
 | ------|------|--------|---------|-----|------|-----|
 """
        results = parse_llama_bench_output(sample_output)
        assert results == [], "Output with no data should return empty list"

    def test_parse_llama_bench_output_malformed(self) -> None:
        """Test parsing malformed output handles errors gracefully."""
        sample_output = """
 | model | size | params | backend | ngl | test | t/s |
 | bad line without proper format
 | test_model | invalid | data | CPU | 0 | pp512 | not_a_number |
 """
        results = parse_llama_bench_output(sample_output)
        # Should handle gracefully and return empty or partial results
        assert isinstance(results, list), "Should return a list even for bad input"
