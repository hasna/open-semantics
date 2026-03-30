# SemHex — Semantic Hexadecimal Encoding

## What This Is

A universal compact discrete encoding for meaning — like hex codes for colors, but for semantics. Text goes in, compact hex codes come out. Codes can be decoded back to meaning, blended (code arithmetic), and compared (semantic distance).

## Architecture

```
Text → Embedding Model → Vector → Codebook Lookup → SemHex Code ($XX.XXXX)
SemHex Code → Codebook Lookup → Centroid → Nearest Concepts → Decoded Output
```

## Key Concepts

- **Codebook**: A frozen lookup table of 65K points in meaning space. Each point has a hex address. Permanent — like Unicode code points.
- **Encoder**: Embeds text, finds nearest codebook entry. Encodes at MEANING level (whole sentences/clauses), not word level.
- **Decoder**: Looks up codebook entry, returns concept labels and neighbors. v0.1 returns structured output; v0.2 will use LLM expansion.
- **Code Format**: `$XX.XXXX.XXXXXX` — Level 1 (coarse, 256 categories) . Level 2 (fine, 65K meanings) . Level 3 (precise)

## Tech Stack

- Python 3.10+
- sentence-transformers (local embeddings, no API key needed)
- scikit-learn (KMeans for codebook training)
- numpy (vector math)
- faiss-cpu (fast nearest-neighbor search)
- click + rich (CLI)

## Commands

```bash
pip install -e ".[all]"           # Install with all deps
semhex encode "text"              # Encode text to SemHex codes
semhex decode "$XX.XXXX"          # Decode codes to meaning
semhex distance "$A" "$B"         # Semantic distance
semhex blend "$A" "$B"            # Code arithmetic
semhex eval roundtrip             # Run roundtrip evaluation
semhex eval composition           # Run composition test
pytest                            # Run all tests
```

## Project Structure

```
semhex/
  core/
    format.py      # $XX.XXXX parsing, validation
    codebook.py    # Load and query frozen codebook
    encoder.py     # Text → SemHex codes
    decoder.py     # SemHex codes → meaning
    distance.py    # Semantic distance between codes
    blend.py       # Code arithmetic
  embeddings/
    base.py        # Abstract embedding interface
    local_embed.py # sentence-transformers adapter
    openai_embed.py# OpenAI adapter
    mock.py        # Mock for tests
  cli.py           # CLI entry point
  mcp_server.py    # MCP server
codebooks/
  v0.1/            # Frozen codebook (numpy arrays + labels)
training/          # Codebook training scripts
evaluation/        # Eval scripts + results
tests/             # pytest test suite
```

## Design Principles

1. Codebook is FROZEN once trained. Codes are permanent addresses.
2. Encode at MEANING level, not word level. Compression comes from semantic chunking.
3. Codes carry more meaning per token than natural language.
4. Nearby codes = nearby meanings (topological signs).
5. Multiple codes per meaning allowed (degeneracy, like DNA codons).
6. Decoder handles composition — codes encode features, decoder binds them.
