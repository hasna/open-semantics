"""SemHex — Semantic Hexadecimal Encoding.

A universal compact discrete encoding for meaning, like hex codes for colors.
"""

__version__ = "0.1.0"

from semhex.core.format import SemHexCode, parse_code, format_code
from semhex.core.codebook import Codebook, load_codebook
from semhex.core.encoder import encode, encode_batch
from semhex.core.decoder import decode
from semhex.core.distance import distance
from semhex.core.blend import blend

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
