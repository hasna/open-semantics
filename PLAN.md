# SemHex — The Plan

## The Big Idea

SemHex is a NEW LANGUAGE. Not a compression algorithm — a language.

An LLM already knows English, French, Mandarin, Arabic. You can ask it to "speak French" and it does. SemHex is teaching it one more language — except this language is made of hex codes instead of words, and it's mathematically structured so that similar meanings have similar codes.

Once the LLM knows SemHex, you just say "respond in SemHex" and it does. Then you decode the output on the other end. The LLM doesn't need special architecture — it just needs to learn the language, like it learned every other language. The map IS the dictionary of this language.

The difference from other languages: SemHex is COMPACT (one code = one phrase), UNIVERSAL (works across all human languages), and STRUCTURED (similar meanings have similar codes, shared prefix = shared meaning).

## Goal

Build the map (the dictionary of SemHex), verify it works with high accuracy, then export it so any LLM can learn this language.

## The Pipeline

```
USER: "I'm frustrated with this bug"
  ↓
ENCODER: embed → find nearest region in the MAP → code $F1.1C3D.42ABCC...
  ↓
LLM: receives code(s) as input, thinks, outputs code(s)
  ↓
DECODER: look up code in the MAP → expand to text
  ↓
OUTPUT: "Have you tried checking the error log?"
```

## The Map

A file (numpy + JSON) containing:
- 65K+ regions of meaning space
- Each region has: a hex code, a centroid vector, example sentences
- ANY LLM can load this map and use it
- Exportable, versioned, frozen once trained

Built using:
1. Embed millions of sentences (OpenAI Matryoshka at 64 dims)
2. Cluster into 65K regions (KMeans)
3. Quantize centroids to hex codes (geohash-style: PCA → multi-bit quantization → hex)
4. Store as codebook file

## The Encoding (geohash for meaning)

```
text → OpenAI embed(text, dims=64) → 64 floats → quantize each dim → pack into hex
```

Format: $XX.XXXX.XXXXXX (semantic → specific)
- First chars = broad meaning area
- More chars = more precise
- Shared prefix = similar meaning
- No lookup table needed for encoding — the code IS the quantized vector

## Accuracy Results So Far

| Config | Bits | Hex Chars | Reconstruction Similarity |
|--------|------|-----------|--------------------------|
| v1: PCA 48d × 1bit | 48 | 12 | 0.45 |
| v2: PCA 48d × 2bit | 96 | 24 | 0.53 |
| v2: PCA 128d × 2bit | 256 | 64 | 0.67 |
| Matryoshka 64d × 2bit | 128 | 32 | 0.94 |
| **Matryoshka 64d × 4bit** | **256** | **64** | **0.991** |

Key insight: use OpenAI's native Matryoshka (dims=64) instead of PCA.

### Best Config: 64d × 4bit (VALIDATED)

- **Mean similarity: 0.991** on 1000 holdout sentences
- 100% above 0.90
- 97.2% above 0.98
- 73.6% above 0.99
- Worst case: 0.943
- 64 hex characters per sentence

## Next Steps

### Phase 1: Perfect the Map (NOW)
- [ ] Train 64d × 4bit quantizer → 256 bits, 64 hex chars → target 0.97+ similarity
- [ ] Scale map to 65K regions using all 89K training sentences
- [ ] Export map as standalone file (numpy + JSON, ~1GB)
- [ ] Verify: encode → decode → measure similarity on 1000 holdout sentences
- [ ] Build map browser CLI: `semhex map search "anger"` → shows nearby codes

### Phase 2: Train Encoder/Decoder Models
- [ ] Generate 10K+ training pairs (text → code, code → text) via Cerebras
- [ ] Fine-tune encoder via brains MCP (gpt-4o-mini → text-to-codes)
- [ ] Fine-tune decoder via brains MCP (gpt-4o-mini → codes-to-text)
- [ ] Measure: fine-tuned encoder consistency (same input → same code?)
- [ ] Measure: fine-tuned decoder accuracy (code → correct text?)

### Phase 3: Train LLM to Think in SemHex
- [ ] Encode conversation dataset into SemHex (both sides)
- [ ] Fine-tune a model that receives AND outputs SemHex codes
- [ ] Add SemHex codes to model vocabulary (each code = 1 token)
- [ ] Measure: tokens saved, latency reduction, cost reduction

### Phase 4: Ship It
- [ ] Package map as downloadable file
- [ ] CLI: `semhex encode/decode/compress/decompress`
- [ ] MCP server for AI agents
- [ ] npm/pip packages
- [ ] Documentation + examples

## Key Principle

The MAP is the product. Not the model, not the code — the MAP. It's like Unicode: a universal lookup table that any system can use. Build the best map possible, export it, and let every LLM benefit.

## Scaling Law

```
similarity = -0.059 + 0.130 × ln(bits)  (R² = 0.994)
```

More bits = more accuracy. Matryoshka embeddings dramatically improve the constant factor.
