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


def _get_codebook():
    from semhex.core.codebook import load_codebook
    try:
        return load_codebook("v0.1")
    except FileNotFoundError:
        console.print("[red]Codebook v0.1 not found. Run:[/red]")
        console.print("  python -m training.build_codebook")
        sys.exit(1)


def _get_provider():
    from semhex.embeddings import get_provider
    return get_provider("auto")


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
    provider = _get_provider()
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
    provider = _get_provider()

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
