"""SemHex MCP Server — expose encoding/decoding as MCP tools.

Tools:
- semhex_encode: Text → SemHex codes
- semhex_decode: SemHex codes → meaning labels
- semhex_distance: Distance between two codes
- semhex_blend: Code arithmetic
- semhex_inspect: Code details + neighbors
- semhex_roundtrip: Encode → decode → show both + similarity

Run: python -m semhex.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "semhex",
    instructions="SemHex — Semantic Hexadecimal Encoding. Encode meaning into compact hex codes, decode them back, measure semantic distance, and perform code arithmetic.",
)

# Lazy-loaded singletons
_codebook = None
_provider = None


def _get_codebook():
    global _codebook
    if _codebook is None:
        from semhex.core.codebook import load_codebook
        _codebook = load_codebook("v0.1")
    return _codebook


def _get_provider():
    global _provider
    if _provider is None:
        from semhex.embeddings import get_provider
        _provider = get_provider("auto")
    return _provider


@mcp.tool()
def semhex_encode(text: str, depth: int = 2) -> dict:
    """Encode text into SemHex codes. Compression happens at meaning level — a sentence becomes a few codes, not one per word.

    Args:
        text: Input text to encode.
        depth: Code depth (1=coarse 256 categories, 2=fine 65K meanings).
    """
    from semhex.core.encoder import encode

    result = encode(text, depth=depth, codebook=_get_codebook(), provider=_get_provider())
    return {
        "codes": result.code_strings,
        "chunks": result.chunks,
        "distances": [round(d, 4) for d in result.distances],
        "compression_ratio": round(result.compression_ratio, 1),
    }


@mcp.tool()
def semhex_decode(codes: str, k_neighbors: int = 3) -> dict:
    """Decode SemHex codes into human-readable meaning labels.

    Args:
        codes: Space-separated SemHex codes (e.g., "$3A.C8F0 $72.B1A0").
        k_neighbors: Number of neighbor concepts to include.
    """
    from semhex.core.decoder import decode

    result = decode(codes, codebook=_get_codebook(), k_neighbors=k_neighbors)
    return result.to_dict()


@mcp.tool()
def semhex_distance(code_a: str, code_b: str) -> dict:
    """Compute semantic distance between two SemHex codes.

    Args:
        code_a: First SemHex code (e.g., "$8A.2100").
        code_b: Second SemHex code (e.g., "$8A.2400").
    """
    from semhex.core.distance import distance, similarity

    cb = _get_codebook()
    d = distance(code_a, code_b, codebook=cb)
    s = similarity(code_a, code_b, codebook=cb)
    return {
        "code_a": code_a,
        "code_b": code_b,
        "distance": round(d, 4),
        "similarity": round(s, 4),
        "interpretation": "identical" if d < 0.05 else "very close" if d < 0.3 else "related" if d < 0.7 else "different" if d < 1.2 else "very different",
    }


@mcp.tool()
def semhex_blend(code_a: str, code_b: str, weight: float = 0.5) -> dict:
    """Blend two SemHex codes via semantic arithmetic. Like color mixing but for meaning.

    Args:
        code_a: First SemHex code.
        code_b: Second SemHex code.
        weight: Weight for first code (0.0-1.0). Default 0.5 = equal blend.
    """
    from semhex.core.blend import blend
    from semhex.core.decoder import decode

    cb = _get_codebook()
    result = blend(code_a, code_b, weight=weight, codebook=cb)

    dec_a = decode([code_a], codebook=cb)
    dec_b = decode([code_b], codebook=cb)
    dec_r = decode([result], codebook=cb)

    return {
        "input_a": {"code": code_a, "label": dec_a.decoded[0].label if dec_a.decoded else "?"},
        "input_b": {"code": code_b, "label": dec_b.decoded[0].label if dec_b.decoded else "?"},
        "result": {"code": str(result), "label": dec_r.decoded[0].label if dec_r.decoded else "?"},
        "weight": weight,
    }


@mcp.tool()
def semhex_inspect(code: str, k_neighbors: int = 5) -> dict:
    """Inspect a SemHex code — show its meaning, category, examples, and neighbors.

    Args:
        code: SemHex code to inspect (e.g., "$8A.2100").
        k_neighbors: Number of neighbors to show.
    """
    from semhex.core.decoder import decode

    result = decode([code], codebook=_get_codebook(), k_neighbors=k_neighbors)
    if not result.decoded:
        return {"error": f"Code {code} not found in codebook"}

    d = result.decoded[0]
    return {
        "code": str(d.code),
        "label": d.label,
        "category": d.l1_label,
        "depth": d.code.depth,
        "examples": d.examples,
        "neighbors": d.neighbors,
    }


@mcp.tool()
def semhex_roundtrip(text: str, depth: int = 2) -> dict:
    """Encode text to SemHex codes, then decode back — shows both sides and compression ratio.

    Args:
        text: Input text to roundtrip.
        depth: Code depth (1=coarse, 2=fine).
    """
    from semhex.core.encoder import encode
    from semhex.core.decoder import decode

    cb = _get_codebook()
    provider = _get_provider()

    enc = encode(text, depth=depth, codebook=cb, provider=provider)
    dec = decode(enc.codes, codebook=cb)

    return {
        "input": text,
        "codes": enc.code_strings,
        "decoded_summary": dec.summary,
        "compression_ratio": round(enc.compression_ratio, 1),
        "n_words": sum(len(c.split()) for c in enc.chunks),
        "n_codes": len(enc.codes),
        "details": [
            {
                "chunk": chunk,
                "code": code_str,
                "label": d.label,
                "distance": round(dist, 4),
            }
            for chunk, code_str, d, dist in zip(enc.chunks, enc.code_strings, dec.decoded, enc.distances)
        ],
    }


@mcp.tool()
def semhex_codebook_info() -> dict:
    """Show codebook statistics — version, dimensions, cluster counts."""
    cb = _get_codebook()
    return {
        "version": cb.version,
        "dimensions": cb.dimensions,
        "l1_clusters": cb.n_level1,
        "l2_clusters": cb.n_level2,
        "total_codes": cb.n_level1 + cb.n_level2,
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
