"""SemHex — Semantic Hexadecimal Encoding.

A universal compact discrete encoding for meaning, like hex codes for colors.
"""

__version__ = "0.1.0"

# Core format (always available, no ML deps)
from semhex.core.format import SemHexCode, parse_code, format_code


def __getattr__(name: str):
    """Lazy imports for modules that need ML dependencies."""
    if name == "Codebook" or name == "load_codebook":
        from semhex.core.codebook import Codebook, load_codebook
        return Codebook if name == "Codebook" else load_codebook
    if name == "encode" or name == "encode_batch":
        from semhex.core.encoder import encode, encode_batch
        return encode if name == "encode" else encode_batch
    if name == "decode":
        from semhex.core.decoder import decode
        return decode
    if name == "distance":
        from semhex.core.distance import distance
        return distance
    if name == "blend":
        from semhex.core.blend import blend
        return blend
    raise AttributeError(f"module 'semhex' has no attribute {name!r}")


__all__ = [
    "SemHexCode",
    "parse_code",
    "format_code",
    "Codebook",
    "load_codebook",
    "encode",
    "encode_batch",
    "decode",
    "distance",
    "blend",
]
