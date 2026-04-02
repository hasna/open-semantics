"""Tests for MCP server tool functions — mocked codebook/provider, no API calls."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from semhex.core.codebook import load_codebook
from semhex.embeddings.mock import MockEmbeddingProvider


@pytest.fixture
def test_cb():
    """Load the test codebook (64d, fast)."""
    return load_codebook("v0.1")


@pytest.fixture
def mock_provider():
    return MockEmbeddingProvider(dimensions=64)


class TestSemhexCodebookInfo:
    def test_returns_dict(self, test_cb):
        from semhex.mcp_server import semhex_codebook_info
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_codebook_info()
        assert isinstance(result, dict)

    def test_has_required_keys(self, test_cb):
        from semhex.mcp_server import semhex_codebook_info
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_codebook_info()
        assert "version" in result
        assert "dimensions" in result
        assert "l1_clusters" in result
        assert "l2_clusters" in result
        assert "total_codes" in result

    def test_total_codes_is_sum(self, test_cb):
        from semhex.mcp_server import semhex_codebook_info
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_codebook_info()
        assert result["total_codes"] == result["l1_clusters"] + result["l2_clusters"]

    def test_get_codebook_falls_back_to_test_when_v01_missing(self):
        from semhex import mcp_server

        mcp_server._codebook = None
        sentinel = object()
        with patch("semhex.core.codebook.load_codebook", side_effect=[FileNotFoundError("missing"), sentinel]) as mock_load:
            result = mcp_server._get_codebook()

        assert result is sentinel
        assert mock_load.call_args_list[0].args == ("v0.1",)
        assert mock_load.call_args_list[1].args == ("test",)
        mcp_server._codebook = None


class TestSemhexDistance:
    def test_returns_dict(self, test_cb):
        from semhex.mcp_server import semhex_distance
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_distance("$00.0000", "$00.0000")
        assert isinstance(result, dict)

    def test_identical_codes_distance_zero(self, test_cb):
        from semhex.mcp_server import semhex_distance
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_distance("$00.0000", "$00.0000")
        assert result["distance"] == 0.0
        assert result["similarity"] == 1.0

    def test_has_interpretation(self, test_cb):
        from semhex.mcp_server import semhex_distance
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_distance("$00.0000", "$00.0000")
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)

    def test_codes_preserved(self, test_cb):
        from semhex.mcp_server import semhex_distance
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_distance("$00.0000", "$01.0000")
        assert result["code_a"] == "$00.0000"
        assert result["code_b"] == "$01.0000"


class TestSemhexBlend:
    def test_returns_dict(self, test_cb):
        from semhex.mcp_server import semhex_blend
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_blend("$00.0000", "$01.0000", weight=0.5)
        assert isinstance(result, dict)

    def test_has_result_code(self, test_cb):
        from semhex.mcp_server import semhex_blend
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_blend("$00.0000", "$01.0000", weight=0.5)
        assert "result" in result

    def test_weight_preserved(self, test_cb):
        from semhex.mcp_server import semhex_blend
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_blend("$00.0000", "$01.0000", weight=0.3)
        assert result["weight"] == 0.3


class TestSemhexEncode:
    def test_returns_dict(self, test_cb, mock_provider):
        from semhex.mcp_server import semhex_encode
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb), \
             patch("semhex.mcp_server._get_provider", return_value=mock_provider):
            result = semhex_encode("hello world", depth=2)
        assert isinstance(result, dict)

    def test_has_codes_list(self, test_cb, mock_provider):
        from semhex.mcp_server import semhex_encode
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb), \
             patch("semhex.mcp_server._get_provider", return_value=mock_provider):
            result = semhex_encode("hello world")
        assert "codes" in result
        assert isinstance(result["codes"], list)
        assert len(result["codes"]) > 0

    def test_has_compression_ratio(self, test_cb, mock_provider):
        from semhex.mcp_server import semhex_encode
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb), \
             patch("semhex.mcp_server._get_provider", return_value=mock_provider):
            result = semhex_encode("hello world this is a test")
        assert "compression_ratio" in result
        assert result["compression_ratio"] > 0


class TestSemhexDecode:
    def test_returns_dict(self, test_cb):
        from semhex.mcp_server import semhex_decode
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_decode("$00.0000")
        assert isinstance(result, dict)

    def test_has_summary(self, test_cb):
        from semhex.mcp_server import semhex_decode
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_decode("$00.0000")
        assert "summary" in result


class TestSemhexInspect:
    def test_returns_dict(self, test_cb):
        from semhex.mcp_server import semhex_inspect
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_inspect("$00.0000")
        assert isinstance(result, dict)

    def test_has_code(self, test_cb):
        from semhex.mcp_server import semhex_inspect
        with patch("semhex.mcp_server._get_codebook", return_value=test_cb):
            result = semhex_inspect("$00.0000")
        assert "code" in result or "label" in result


class TestSemhexCompress:
    @patch("semhex.core.codec._get_cerebras")
    def test_returns_dict(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="FRU.BUG.HLP"))]
        )
        mock_get.return_value = client

        from semhex.mcp_server import semhex_compress
        result = semhex_compress("I'm frustrated with this bug", quality=2, provider="cerebras")
        assert isinstance(result, dict)

    @patch("semhex.core.codec._get_cerebras")
    def test_has_codes(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="FRU.BUG.HLP"))]
        )
        mock_get.return_value = client

        from semhex.mcp_server import semhex_compress
        result = semhex_compress("test text", quality=2, provider="cerebras")
        assert "codes" in result
        assert "compression_ratio" in result
        assert result["compression_ratio"] > 0

    @patch("semhex.core.codec._get_cerebras")
    def test_has_char_counts(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="X.Y"))]
        )
        mock_get.return_value = client

        from semhex.mcp_server import semhex_compress
        result = semhex_compress("hello world", quality=2, provider="cerebras")
        assert "input_chars" in result
        assert "code_chars" in result
        assert result["input_chars"] == len("hello world")


class TestSemhexDecompress:
    @patch("semhex.core.codec._get_cerebras")
    def test_returns_dict(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="I am frustrated with this bug"))]
        )
        mock_get.return_value = client

        from semhex.mcp_server import semhex_decompress
        result = semhex_decompress("FRU.BUG", provider="cerebras")
        assert isinstance(result, dict)
        assert "text" in result
        assert "codes" in result


class TestSemhexDictEncode:
    def test_returns_dict(self):
        from semhex.mcp_server import semhex_dict_encode
        result = semhex_dict_encode("hello world")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from semhex.mcp_server import semhex_dict_encode
        result = semhex_dict_encode("hello world")
        for key in ("text", "codes", "compression_ratio", "input_chars", "code_chars"):
            assert key in result

    def test_text_preserved(self):
        from semhex.mcp_server import semhex_dict_encode
        result = semhex_dict_encode("hello world")
        assert result["text"] == "hello world"

    def test_codes_non_empty(self):
        from semhex.mcp_server import semhex_dict_encode
        result = semhex_dict_encode("hello world")
        assert len(result["codes"]) > 0

    def test_char_counts_consistent(self):
        from semhex.mcp_server import semhex_dict_encode
        text = "I am frustrated with this bug"
        result = semhex_dict_encode(text)
        assert result["input_chars"] == len(text)
        assert result["code_chars"] == len(result["codes"])


class TestSemhexDictDecode:
    def test_returns_dict(self):
        from semhex.mcp_server import semhex_dict_decode
        result = semhex_dict_decode("00.01")
        assert isinstance(result, dict)

    def test_has_text_and_codes(self):
        from semhex.mcp_server import semhex_dict_decode
        result = semhex_dict_decode("00.01")
        assert "text" in result
        assert "codes" in result

    def test_detailed_flag(self):
        from semhex.mcp_server import semhex_dict_decode
        result = semhex_dict_decode("00.01", detailed=True)
        assert "entries" in result
        assert "n_found" in result

    def test_roundtrip(self):
        from semhex.mcp_server import semhex_dict_encode, semhex_dict_decode
        enc = semhex_dict_encode("I am frustrated with this bug")
        dec = semhex_dict_decode(enc["codes"])
        assert "frustrated" in dec["text"]


class TestSemhexRgbDecode:
    def test_returns_dict(self):
        from semhex.mcp_server import semhex_rgb_decode
        result = semhex_rgb_decode("$00.00.00")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from semhex.mcp_server import semhex_rgb_decode
        result = semhex_rgb_decode("$00.00.00")
        assert "code" in result
        assert "description" in result
        assert "dimensions" in result

    def test_dimensions_has_labels(self):
        from semhex.mcp_server import semhex_rgb_decode
        result = semhex_rgb_decode("$00.00.00")
        dims = result["dimensions"]
        assert "domain_label" in dims
        assert "agent_label" in dims
        assert "intent_label" in dims

    def test_technology_domain(self):
        from semhex.mcp_server import semhex_rgb_decode
        # domain=8 → technology; construct a code with domain=8
        from semhex.core.semantic_rgb import SemanticColor
        c = SemanticColor(8, 4, 4, 0, 8, 0, 8)
        result = semhex_rgb_decode(c.to_hex())
        assert result["dimensions"]["domain_label"] == "technology"

    def test_code_preserved(self):
        from semhex.mcp_server import semhex_rgb_decode
        result = semhex_rgb_decode("$00.00.00")
        assert result["code"] == "$00.00.00"
