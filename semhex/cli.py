"""SemHex CLI — encode, decode, distance, blend, inspect, roundtrip.

Usage:
    semhex encode "Can you help me debug this?"
    semhex decode "$3A.C8F0 $72.B1A0"
    semhex distance "$8A.2100" "$8A.2400"
    semhex blend "$8A.2100" "$60.3000"
    semhex inspect "$8A.2100"
    semhex roundtrip "The cat sat on the mat."
    semhex codebook info
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def _get_codebook(version: str = "v0.1"):
    from semhex.core.codebook import load_codebook
    from pathlib import Path
    import numpy as np

    # Check if the requested version has the required numpy arrays
    base = Path(__file__).parent.parent / "codebooks" / version
    has_npy = (base / "level1.npy").exists()

    if not has_npy and version == "v0.1":
        # v0.1 has labels but missing .npy centroids — fall back to test codebook
        err_console.print(f"[yellow]Note: codebook {version} missing centroid arrays, using test codebook.[/yellow]")
        err_console.print("[dim]Run 'python -m training.build_codebook' to generate the full codebook.[/dim]")
        try:
            return load_codebook("test")
        except FileNotFoundError:
            err_console.print("[red]No usable codebook found.[/red]")
            sys.exit(1)

    try:
        return load_codebook(version)
    except FileNotFoundError:
        err_console.print(f"[red]Codebook {version} not found. Run:[/red]")
        err_console.print("  python -m training.build_codebook")
        sys.exit(1)


def _get_provider(codebook=None):
    from semhex.embeddings import get_provider
    provider = get_provider("auto")
    # If codebook dimensions don't match the provider, fall back to mock
    if codebook is not None and provider.dimensions != codebook.dimensions:
        from semhex.embeddings.mock import MockEmbeddingProvider
        err_console.print(f"[yellow]Note: provider dims ({provider.dimensions}) != codebook dims ({codebook.dimensions}), using mock provider.[/yellow]")
        return MockEmbeddingProvider(dimensions=codebook.dimensions)
    return provider


@click.group()
@click.version_option(version="0.1.0", prog_name="semhex")
def main():
    """SemHex — Semantic Hexadecimal Encoding."""
    pass


@main.command()
@click.argument("text")
@click.option("--depth", "-d", default=2, type=int, help="Code depth: 1=coarse, 2=fine")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def encode(text: str, depth: int, json_output: bool):
    """Encode text into SemHex codes."""
    from semhex.core.encoder import encode as do_encode

    codebook = _get_codebook()
    provider = _get_provider(codebook)
    result = do_encode(text, depth=depth, codebook=codebook, provider=provider)

    if json_output:
        click.echo(json.dumps({
            "codes": result.code_strings,
            "chunks": result.chunks,
            "distances": result.distances,
            "compression_ratio": result.compression_ratio,
        }, indent=2))
        return

    table = Table(title="SemHex Encoding")
    table.add_column("Chunk", style="dim")
    table.add_column("Code", style="bold cyan")
    table.add_column("Distance", justify="right")

    for chunk, code, dist in zip(result.chunks, result.code_strings, result.distances):
        table.add_row(
            chunk[:60] + "..." if len(chunk) > 60 else chunk,
            code,
            f"{dist:.4f}",
        )

    console.print(table)
    console.print(f"\n[bold]Codes:[/bold] {' '.join(result.code_strings)}")
    console.print(f"[bold]Compression:[/bold] {result.compression_ratio:.1f}x ({len(result.chunks)} codes for {sum(len(c.split()) for c in result.chunks)} words)")


@main.command()
@click.argument("codes")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def decode(codes: str, json_output: bool):
    """Decode SemHex codes to meaning."""
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    result = do_decode(codes, codebook=codebook)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    table = Table(title="SemHex Decoding")
    table.add_column("Code", style="bold cyan")
    table.add_column("Label", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Examples", style="dim")

    for d in result.decoded:
        table.add_row(
            str(d.code),
            d.label,
            d.l1_label,
            ", ".join(d.examples[:3]) if d.examples else "-",
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {result.summary}")


@main.command()
@click.argument("code_a")
@click.argument("code_b")
def distance(code_a: str, code_b: str):
    """Compute semantic distance between two codes."""
    from semhex.core.distance import distance as do_distance, similarity as do_similarity

    codebook = _get_codebook()
    d = do_distance(code_a, code_b, codebook=codebook)
    s = do_similarity(code_a, code_b, codebook=codebook)

    console.print(f"[bold]{code_a}[/bold] ↔ [bold]{code_b}[/bold]")
    console.print(f"  Distance:   [cyan]{d:.4f}[/cyan] (0=identical, 2=opposite)")
    console.print(f"  Similarity: [cyan]{s:.4f}[/cyan] (1=identical, -1=opposite)")


@main.command()
@click.argument("code_a")
@click.argument("code_b")
@click.option("--weight", "-w", default=0.5, type=float, help="Weight for first code (0-1)")
def blend(code_a: str, code_b: str, weight: float):
    """Blend two codes via semantic arithmetic."""
    from semhex.core.blend import blend as do_blend
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    result = do_blend(code_a, code_b, weight=weight, codebook=codebook)

    # Decode all three for display
    dec_a = do_decode([code_a], codebook=codebook)
    dec_b = do_decode([code_b], codebook=codebook)
    dec_r = do_decode([result], codebook=codebook)

    label_a = dec_a.decoded[0].label if dec_a.decoded else "?"
    label_b = dec_b.decoded[0].label if dec_b.decoded else "?"
    label_r = dec_r.decoded[0].label if dec_r.decoded else "?"

    console.print(f"[bold]{code_a}[/bold] ({label_a}) + [bold]{code_b}[/bold] ({label_b})")
    console.print(f"  = [bold cyan]{result}[/bold cyan] ({label_r})")
    console.print(f"  weight: {weight:.2f} / {1 - weight:.2f}")


@main.command()
@click.argument("code")
def inspect(code: str):
    """Inspect a SemHex code — show centroid details and neighbors."""
    from semhex.core.format import parse_code
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    parsed = parse_code(code)
    result = do_decode([parsed], codebook=codebook, k_neighbors=5)

    if not result.decoded:
        console.print("[red]Code not found in codebook[/red]")
        return

    d = result.decoded[0]
    console.print(f"[bold]Code:[/bold] {d.code}")
    console.print(f"[bold]Label:[/bold] {d.label}")
    console.print(f"[bold]Category:[/bold] {d.l1_label}")
    console.print(f"[bold]Depth:[/bold] {d.code.depth}")
    if d.examples:
        console.print(f"[bold]Examples:[/bold] {', '.join(d.examples)}")
    if d.neighbors:
        console.print(f"[bold]Neighbors:[/bold]")
        for n in d.neighbors:
            console.print(f"  {n}")


@main.command()
@click.argument("text")
@click.option("--depth", "-d", default=2, type=int)
def roundtrip(text: str, depth: int):
    """Encode text, then decode the codes — show both sides."""
    from semhex.core.encoder import encode as do_encode
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    provider = _get_provider(codebook)

    # Encode
    enc = do_encode(text, depth=depth, codebook=codebook, provider=provider)
    # Decode
    dec = do_decode(enc.codes, codebook=codebook)

    console.print(f"[bold]Input:[/bold] {text}")
    console.print(f"[bold]Codes:[/bold] {' '.join(enc.code_strings)}")
    console.print(f"[bold]Decoded:[/bold] {dec.summary}")
    console.print(f"[bold]Compression:[/bold] {enc.compression_ratio:.1f}x")

    # Detail table
    table = Table(title="Roundtrip Detail")
    table.add_column("Chunk", style="dim")
    table.add_column("Code", style="cyan bold")
    table.add_column("Decoded Label")
    table.add_column("Distance", justify="right")

    for chunk, code_str, d, dist in zip(enc.chunks, enc.code_strings, dec.decoded, enc.distances):
        table.add_row(
            chunk[:50] + "..." if len(chunk) > 50 else chunk,
            code_str,
            d.label,
            f"{dist:.4f}",
        )

    console.print(table)


@main.command(name="hash")
@click.argument("text")
@click.option("--bits", "-b", default=4, type=int, help="Bits per dimension: 2 (compact) or 4 (precise)")
def hash_cmd(text: str, bits: int):
    """Encode text into a SemHex geohash — a mathematical semantic address."""
    import numpy as np
    from openai import OpenAI
    from semhex.core.geohash_v2 import SemHasher

    client = OpenAI()
    hasher = SemHasher(n_dims=64, bits_per_dim=bits)
    state_name = f"matryoshka_64d_{bits}b" if bits != 2 else "matryoshka_64d_2b_full"
    try:
        hasher.load(state_name)
    except FileNotFoundError:
        console.print(f"[red]Trained state '{state_name}' not found. Run training first.[/red]")
        return

    resp = client.embeddings.create(input=[text], model="text-embedding-3-small", dimensions=64)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    code = hasher.encode(vec)
    hex_chars = hasher.hex_length
    total_bits = hasher.total_bits

    console.print(f"[bold]Input:[/bold] {text}")
    console.print(f"[bold]Code:[/bold]  [cyan]{code}[/cyan]")
    console.print(f"[bold]Bits:[/bold]  {total_bits} ({hex_chars} hex chars)")


@main.command(name="unhash")
@click.argument("code")
@click.option("--bits", "-b", default=4, type=int, help="Bits per dimension used during encoding")
def unhash_cmd(code: str, bits: int):
    """Decode a SemHex geohash back to the nearest concepts from the map."""
    import json
    import numpy as np
    from semhex.core.geohash_v2 import SemHasher

    hasher = SemHasher(n_dims=64, bits_per_dim=bits)
    state_name = f"matryoshka_64d_{bits}b" if bits != 2 else "matryoshka_64d_2b_full"
    try:
        hasher.load(state_name)
    except FileNotFoundError:
        console.print(f"[red]Trained state not found.[/red]")
        return

    # Decode to vector
    vec = hasher.decode(code)

    # Find nearest regions in map
    from pathlib import Path
    map_dir = Path("codebooks/map_v1")
    if not (map_dir / "centroids.npy").exists():
        console.print(f"[red]Map not found at {map_dir}. Run build_map first.[/red]")
        return

    centroids = np.load(map_dir / "centroids.npy").astype(np.float32)
    labels = json.loads((map_dir / "labels.json").read_text())

    # Find nearest centroids
    sims = centroids @ vec
    top5 = np.argsort(-sims)[:5]

    console.print(f"[bold]Code:[/bold] {code}")
    console.print(f"[bold]Nearest regions:[/bold]")
    for idx in top5:
        info = labels.get(str(idx), {})
        examples = info.get("examples", [])
        region_code = info.get("hex_code", "?")
        sim = float(sims[idx])
        ex = examples[0][:80] if examples else "(no examples)"
        console.print(f"  [cyan]{region_code[:25]}...[/cyan] (sim={sim:.4f})")
        console.print(f"    \"{ex}\"")


@main.command(name="compress")
@click.argument("text")
@click.option("--quality", "-q", default=2, type=int, help="Quality: 1 (max compress) to 4 (near-lossless)")
@click.option("--provider", "-p", default="cerebras", help="LLM provider: cerebras or openai")
def compress_cmd(text: str, quality: int, provider: str):
    """Compress text into SemHex codes using LLM."""
    from semhex.core.codec import compress
    codes = compress(text, quality=quality, provider=provider)
    console.print(f"[bold]Input:[/bold]  {text}")
    console.print(f"[bold]Codes:[/bold]  [cyan]{codes}[/cyan]")
    console.print(f"[bold]Ratio:[/bold]  {len(text) / max(len(codes), 1):.1f}x ({len(text)} → {len(codes)} chars)")


@main.command(name="decompress")
@click.argument("codes")
@click.option("--provider", "-p", default="cerebras", help="LLM provider: cerebras or openai")
def decompress_cmd(codes: str, provider: str):
    """Decompress SemHex codes back to text using LLM."""
    from semhex.core.codec import decompress
    text = decompress(codes, provider=provider)
    console.print(f"[bold]Codes:[/bold]  {codes}")
    console.print(f"[bold]Text:[/bold]   {text}")


@main.command(name="codec-roundtrip")
@click.argument("text")
@click.option("--quality", "-q", default=2, type=int, help="Quality: 1 to 4")
@click.option("--provider", "-p", default="cerebras")
def codec_roundtrip_cmd(text: str, quality: int, provider: str):
    """Compress → decompress roundtrip with similarity measurement."""
    from semhex.core.codec import roundtrip as codec_roundtrip
    r = codec_roundtrip(text, quality=quality, provider=provider)
    console.print(f"[bold]Input:[/bold]    {r['input']}")
    console.print(f"[bold]Codes:[/bold]    [cyan]{r['codes']}[/cyan]")
    console.print(f"[bold]Output:[/bold]   {r['output']}")
    console.print(f"[bold]Ratio:[/bold]    {r['compression_ratio']}x")
    if r['semantic_similarity']:
        sim = r['semantic_similarity']
        color = "green" if sim > 0.7 else "yellow" if sim > 0.5 else "red"
        console.print(f"[bold]Similarity:[/bold] [{color}]{sim:.4f}[/{color}]")
    console.print(f"[bold]Time:[/bold]    compress {r['compress_time']}s + decompress {r['decompress_time']}s")


@main.command(name="dict-encode")
@click.argument("text")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def dict_encode_cmd(text: str, json_output: bool):
    """Encode text using the local dictionary (no API key needed, instant)."""
    from semhex.core.dict_encoder import dict_encode
    codes = dict_encode(text)
    ratio = len(text) / max(len(codes), 1)
    if json_output:
        click.echo(json.dumps({"text": text, "codes": codes, "compression_ratio": round(ratio, 2)}, indent=2))
        return
    console.print(f"[bold]Input:[/bold]  {text}")
    console.print(f"[bold]Codes:[/bold]  [cyan]{codes}[/cyan]")
    console.print(f"[bold]Ratio:[/bold]  {ratio:.1f}x ({len(text)} → {len(codes)} chars)")


@main.command(name="dict-decode")
@click.argument("codes")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--detailed", "-d", is_flag=True, help="Show per-code breakdown")
def dict_decode_cmd(codes: str, json_output: bool, detailed: bool):
    """Decode dictionary codes back to text (no API key needed, instant)."""
    from semhex.core.dict_decoder import dict_decode, dict_decode_detailed
    if json_output or detailed:
        result = dict_decode_detailed(codes)
        if json_output:
            click.echo(json.dumps(result, indent=2))
            return
        # detailed table
        table = Table(title="Dictionary Decode")
        table.add_column("Code", style="cyan bold")
        table.add_column("Text")
        table.add_column("Found", justify="center")
        for e in result["entries"]:
            table.add_row(e["code"], e["text"], "✓" if e["found"] else "✗")
        console.print(table)
        console.print(f"\n[bold]Text:[/bold]    {result['text']}")
        console.print(f"[bold]Found:[/bold]   {result['n_found']}/{result['n_codes']} codes")
        return
    text = dict_decode(codes)
    console.print(f"[bold]Codes:[/bold]  {codes}")
    console.print(f"[bold]Text:[/bold]   {text}")


@main.command(name="dict-info")
def dict_info_cmd():
    """Show dictionary statistics."""
    import json as _json
    from pathlib import Path
    dict_path = Path(__file__).parent.parent / "codebooks" / "dictionary_v1.json"
    if not dict_path.exists():
        err_console.print("[red]dictionary_v1.json not found[/red]")
        return
    d = _json.loads(dict_path.read_text())
    console.print(f"[bold]Dictionary version:[/bold] {d.get('version', 'unknown')}")
    console.print(f"[bold]Entries:[/bold]        {d.get('n_words', 0):,}")
    console.print(f"[bold]Phrases:[/bold]        {d.get('n_phrases', 0):,}")
    tiers = d.get("tiers", {})
    for tier, count in tiers.items():
        console.print(f"  [cyan]{tier}:[/cyan] {count:,} codes")


@main.command(name="dict-roundtrip")
@click.argument("text")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def dict_roundtrip_cmd(text: str, json_output: bool):
    """Dictionary encode → decode roundtrip (no API key needed)."""
    from semhex.core.dict_encoder import dict_encode
    from semhex.core.dict_decoder import dict_decode
    codes = dict_encode(text)
    decoded = dict_decode(codes)
    ratio = len(text) / max(len(codes), 1)
    if json_output:
        click.echo(json.dumps({"input": text, "codes": codes, "output": decoded, "compression_ratio": round(ratio, 2)}, indent=2))
        return
    console.print(f"[bold]Input:[/bold]   {text}")
    console.print(f"[bold]Codes:[/bold]   [cyan]{codes}[/cyan]")
    console.print(f"[bold]Output:[/bold]  {decoded}")
    console.print(f"[bold]Ratio:[/bold]   {ratio:.1f}x")


@main.group()
def codebook():
    """Codebook management commands."""
    pass


@codebook.command(name="info")
def codebook_info():
    """Show codebook statistics."""
    cb = _get_codebook()

    console.print(f"[bold]Version:[/bold] {cb.version}")
    console.print(f"[bold]Dimensions:[/bold] {cb.dimensions}")
    console.print(f"[bold]L1 clusters:[/bold] {cb.n_level1}")
    console.print(f"[bold]L2 clusters:[/bold] {cb.n_level2}")
    console.print(f"[bold]Total codes:[/bold] {cb.n_level1 + cb.n_level2:,}")


@main.group()
def eval():
    """Run evaluation suites."""
    pass


@eval.command(name="roundtrip")
def eval_roundtrip_cmd():
    """Evaluate encode→decode roundtrip quality."""
    from evaluation.eval_roundtrip import eval_roundtrip
    result = eval_roundtrip()
    console.print(f"\n[bold]Roundtrip ({result.n_sentences} sentences):[/bold]")
    console.print(f"  Mean similarity: [cyan]{result.mean_similarity:.4f}[/cyan]")
    console.print(f"  Min similarity:  [cyan]{result.min_similarity:.4f}[/cyan]")
    console.print(f"  Std deviation:   {result.std_similarity:.4f}")
    console.print(f"  Time:            {result.elapsed_seconds:.2f}s")


@eval.command(name="composition")
@click.option("--n-pairs", default=200, help="Number of pairs to test")
def eval_composition_cmd(n_pairs: int):
    """Evaluate code arithmetic compositionality (nero's test)."""
    from evaluation.eval_composition import eval_composition
    result = eval_composition(n_pairs=n_pairs)
    console.print(f"\n[bold]Composition ({result.n_pairs} pairs):[/bold]")
    console.print(f"  Validity rate:   [cyan]{result.validity_rate:.1%}[/cyan] (target: >75%)")
    console.print(f"  Mean similarity: [cyan]{result.mean_similarity:.4f}[/cyan]")
    console.print(f"  Time:            {result.elapsed_seconds:.2f}s")


@eval.command(name="distance")
def eval_distance_cmd():
    """Evaluate distance correlation with embedding space."""
    from evaluation.eval_distance import eval_distance_correlation
    result = eval_distance_correlation()
    console.print(f"\n[bold]Distance Correlation ({result.n_pairs} pairs):[/bold]")
    console.print(f"  Spearman r:  [cyan]{result.spearman_r:.4f}[/cyan] (target: >0.80)")
    console.print(f"  p-value:     {result.spearman_p:.6f}")
    console.print(f"  Time:        {result.elapsed_seconds:.2f}s")


@eval.command(name="benchmark")
def eval_benchmark_cmd():
    """Run performance benchmark."""
    from evaluation.benchmark import run_benchmark
    result = run_benchmark()
    console.print(f"\n[bold]Benchmark ({result.n_sentences} sentences):[/bold]")
    console.print(f"  Compression:    [cyan]{result.compression_ratio:.1f}x[/cyan]")
    console.print(f"  Encode speed:   [cyan]{result.encode_rate:.0f}[/cyan] sentences/sec")
    console.print(f"  Lookup latency: [cyan]{result.lookup_time * 1000:.3f}ms[/cyan]")
    console.print(f"  Codebook RAM:   {result.codebook_memory_mb:.2f} MB")


@eval.command(name="all")
@click.option("--n-pairs", default=200, help="Number of composition pairs")
def eval_all_cmd(n_pairs: int):
    """Run all evaluations."""
    import json as json_mod
    from evaluation.eval_roundtrip import eval_roundtrip
    from evaluation.eval_composition import eval_composition
    from evaluation.eval_distance import eval_distance_correlation
    from evaluation.benchmark import run_benchmark

    console.print("[bold]Running all evaluations...[/bold]\n")

    rt = eval_roundtrip()
    console.print(f"[green]Roundtrip:[/green] mean_sim={rt.mean_similarity:.4f} min={rt.min_similarity:.4f}")

    comp = eval_composition(n_pairs=n_pairs)
    console.print(f"[green]Composition:[/green] validity={comp.validity_rate:.1%} mean_sim={comp.mean_similarity:.4f}")

    dist = eval_distance_correlation()
    console.print(f"[green]Distance:[/green] spearman_r={dist.spearman_r:.4f} p={dist.spearman_p:.4f}")

    bench = run_benchmark()
    console.print(f"[green]Benchmark:[/green] {bench.compression_ratio:.1f}x compression, {bench.encode_rate:.0f} sent/s")

    console.print("\n[bold]All evaluations complete.[/bold]")


if __name__ == "__main__":
    main()
