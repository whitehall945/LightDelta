"""Summarize measured data; no target speedups or fabricated missing rows."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    operator = json.loads((args.results / "operator.json").read_text())
    service = args.results / "service"
    paths = sorted(service.glob("round-*/*/i*-o*-c*.json"))
    groups = defaultdict(list)
    for path in paths:
        row = json.loads(path.read_text())
        key = (
            row["config"],
            row["records"][0]["input_tokens"],
            row["records"][0]["output_tokens"],
            row["concurrency"],
        )
        groups[key].append(row)
    measured = []
    for (config, inputs, outputs, concurrency), runs in groups.items():
        records = [r for run in runs for r in run["records"]]
        speculation = [
            r["metrics"]["speculative_decoding"]
            for r in records
            if r.get("metrics") and r["metrics"].get("speculative_decoding")
        ]
        steps = sum(s["num_spec_steps"] for s in speculation)
        accepted = sum(s["num_accepted_draft_tokens"] for s in speculation)
        drafted = sum(s["num_draft_tokens"] for s in speculation)
        measured.append(
            {
                "config": config,
                "input_tokens": inputs,
                "output_tokens": outputs,
                "concurrency": concurrency,
                "runs": len(runs),
                "requests": len(records),
                "output_tokens_per_s": statistics.mean(r["output_tokens_per_s"] for r in runs),
                "throughput_min": min(r["output_tokens_per_s"] for r in runs),
                "throughput_max": max(r["output_tokens_per_s"] for r in runs),
                "ttft_ms": statistics.median(r["ttft_s"] for r in records) * 1000,
                "tpot_ms": statistics.median(r["tpot_s"] for r in records) * 1000,
                "mean_acceptance_length": 1 + accepted / steps if steps else None,
                "draft_acceptance_rate": accepted / drafted if drafted else None,
            }
        )
    memory = {}
    for path in service.glob("round-*/*/memory.csv"):
        peak = max(float(line) for line in path.read_text().splitlines() if line.strip())
        memory[path.parent.name] = max(memory.get(path.parent.name, 0), peak)
    native = json.loads((args.results / "native/control.json").read_text())
    quality = {}
    for path in service.glob("round-*/*/control.json"):
        rows = json.loads(path.read_text())
        quality[str(path.parent)] = {
            "exact_token_matches": sum(
                a["token_ids"] == b["token_ids"] for a, b in zip(native, rows, strict=True)
            ),
            "prompts": len(rows),
        }
    workload_checks = []
    for path in paths:
        baseline = path.parent.parent / "reference" / path.name
        if path.parent.name == "reference" or not baseline.exists():
            continue
        expected = json.loads(baseline.read_text())["records"]
        actual = json.loads(path.read_text())["records"]
        prefix_lengths = []
        for a, b in zip(expected, actual, strict=True):
            left, right = a["token_ids"], b["token_ids"]
            prefix_lengths.append(
                next(
                    (
                        i
                        for i, pair in enumerate(zip(left, right, strict=True))
                        if pair[0] != pair[1]
                    ),
                    len(left),
                )
            )
        workload_checks.append(
            {
                "path": str(path),
                "exact_token_matches": sum(
                    n == a["output_tokens"] for n, a in zip(prefix_lengths, actual, strict=True)
                ),
                "requests": len(actual),
                "mean_matching_prefix_tokens": statistics.mean(prefix_lengths),
            }
        )
    jit_warnings = {}
    for path in service.glob("round-*/*/server.log"):
        timed = False
        warnings = []
        for line in path.read_text().splitlines():
            if "GET /metrics" in line:
                timed = not timed
            if timed and "JIT compilation during inference" in line:
                warnings.append(line)
        jit_warnings[str(path.parent)] = warnings
    summary = {
        "operator": operator,
        "service": measured,
        "peak_memory_mib": memory,
        "generation_checks": quality,
        "workload_token_checks": workload_checks,
        "jit_warnings_in_timed_regions": jit_warnings,
    }
    (args.output / "measurements.json").write_text(json.dumps(summary, indent=2))

    names = ["reference", "fusion", "indexed", "graph", "full"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for batch in (1, 4, 8):
        values = [
            next(
                r["median_us"]
                for r in operator["measurements"]
                if r["config"] == name and r["batch"] == batch and r["tokens_per_request"] == 1
            )
            for name in names[:4]
        ]
        axes[0].plot(names[:4], values, marker="o", label=f"batch {batch}")
    axes[0].set(
        ylabel="GDN latency (microseconds, log scale)",
        yscale="log",
        title="Post-convolution GDN, 1 token/request",
    )
    axes[0].legend()
    for concurrency in (1, 4, 8):
        rows = [
            next(
                (
                    r
                    for r in measured
                    if r["config"] == name
                    and r["input_tokens"] == 512
                    and r["output_tokens"] == 128
                    and r["concurrency"] == concurrency
                ),
                None,
            )
            for name in names
        ]
        if all(row is not None for row in rows):
            axes[1].plot(
                names,
                [r["output_tokens_per_s"] for r in rows],
                marker="o",
                label=f"concurrency {concurrency}",
            )
    axes[1].set(ylabel="Effective output tokens/second", title="Service: 512 input / 128 output")
    axes[1].legend()
    figure.savefig(args.output / "performance.png", dpi=180)
    plt.close(figure)

    lines = [
        "# H100 performance report",
        "",
        "![Measured performance](performance.png)",
        "",
        "All numbers below come from `results/`. Full numeric data is in "
        "[measurements.json](measurements.json).",
        f"The service matrix contains {len(paths)} measured workload runs and "
        f"{sum(r['requests'] for r in measured)} requests. "
        "Hardware, package versions and model identity are recorded in "
        "[environment.json](environment.json) and [model-manifest.json](model-manifest.json).",
        "",
        "## GDN operator",
        "",
        "| Batch | Tokens/request | Reference µs | Fused packed µs | Indexed µs | Graph µs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for batch in (1, 4, 8):
        for width in (1, 2, 3, 4):
            values = [
                next(
                    r["median_us"]
                    for r in operator["measurements"]
                    if r["config"] == name
                    and r["batch"] == batch
                    and r["tokens_per_request"] == width
                )
                for name in names[:4]
            ]
            lines.append(f"| {batch} | {width} | " + " | ".join(f"{x:.2f}" for x in values) + " |")
    lookup = {
        (r["config"], r["input_tokens"], r["output_tokens"], r["concurrency"]): r for r in measured
    }
    lines += [
        "",
        "## Incremental service throughput",
        "",
        "Ratios use the same workload and the mean throughput across rounds. "
        "The denominator is the preceding configuration, except E/A.",
        "",
        "| Input/output | Concurrency | Fusion B/A | Indexed C/B | Graph D/C | "
        "MTP E/D | Full E/A |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    workloads = sorted({key[1:] for key in lookup})
    for workload in workloads:
        if all((name, *workload) in lookup for name in names):
            rates = [lookup[name, *workload]["output_tokens_per_s"] for name in names]
            ratios = [b / a for a, b in zip(rates[:-1], rates[1:], strict=True)]
            ratios.append(rates[-1] / rates[0])
            lines.append(
                f"| {workload[0]}/{workload[1]} | {workload[2]} | "
                + " | ".join(f"{x:.2f}×" for x in ratios)
                + " |"
            )
    variants = ["full_no_graph", "full_packed", "full_prefill", "mtp1", "mtp2"]
    lines += [
        "",
        "## Removal and depth ablations",
        "",
        "Throughput relative to E/full (1.00×). Values below one are slower. "
        "The final column is full TTFT divided by fused-prefill TTFT; above one "
        "means lower TTFT with fused prefill state I/O.",
        "",
        "| Input/output | Concurrency | No graph | Packed states | Fused prefill | "
        "MTP depth 1 | MTP depth 2 | Prefill TTFT ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for workload in workloads:
        if all((name, *workload) in lookup for name in ["full", *variants]):
            full = lookup["full", *workload]
            ratios = [
                lookup[name, *workload]["output_tokens_per_s"] / full["output_tokens_per_s"]
                for name in variants
            ]
            ratios.append(full["ttft_ms"] / lookup["full_prefill", *workload]["ttft_ms"])
            lines.append(
                f"| {workload[0]}/{workload[1]} | {workload[2]} | "
                + " | ".join(f"{x:.2f}×" for x in ratios)
                + " |"
            )
    lines += [
        "",
        "## Streaming service",
        "",
        "TTFT and TPOT are medians across measured requests. Throughput is the mean "
        "of available per-round aggregate rates; a range is shown only for multiple rounds.",
        "",
        "| Configuration | Input/output | Concurrency | Runs | TTFT ms | TPOT ms | "
        "Output tok/s (run range) | Mean acceptance length | Draft acceptance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in measured:
        acceptance = f"{r['mean_acceptance_length']:.2f}" if r["mean_acceptance_length"] else "—"
        draft_rate = (
            f"{r['draft_acceptance_rate']:.1%}" if r["draft_acceptance_rate"] is not None else "—"
        )
        rate = f"{r['output_tokens_per_s']:.2f}"
        if r["runs"] > 1:
            rate += f" ({r['throughput_min']:.2f}–{r['throughput_max']:.2f})"
        lines.append(
            f"| {r['config']} | {r['input_tokens']}/{r['output_tokens']} | "
            f"{r['concurrency']} | {r['runs']} | {r['ttft_ms']:.2f} | "
            f"{r['tpot_ms']:.2f} | {rate} | {acceptance} | {draft_rate} |"
        )
    lines += [
        "",
        "## Memory and generation checks",
        "",
        "| Configuration | Peak device memory MiB |",
        "| --- | --- |",
        *[f"| {name} | {value:.0f} |" for name, value in sorted(memory.items())],
        "",
        f"Native-vLLM regression: {sum(q['exact_token_matches'] for q in quality.values())}/"
        f"{sum(q['prompts'] for q in quality.values())} exact token-sequence matches "
        f"across {len(quality)} server launches. Every suite launch asserts these matches.",
        "",
        "Longer workload token comparisons against A are retained in "
        "`workload_token_checks` in the numeric report. A matching prefix is a "
        "numerical regression observation; it does not measure answer quality.",
        "",
        f"Logged first-use JIT warnings inside timed regions: "
        f"{sum(len(warnings) for warnings in jit_warnings.values())}. "
        "The audit uses each workload's before/after `/metrics` requests in the "
        "server log; it detects logged warnings, not every possible compilation event.",
    ]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "The delivered matrix uses one complete round after the warmup correction. "
        "No statistical confidence interval or 512-token output speedup is claimed.",
        "",
        "A/reference and B/fusion share the exact Torch gather/scatter path. B/A "
        "isolates fused computation; "
        "C/indexed removes packing; D/graph captures decode; E/full enables three MTP candidates. "
        "Removal configurations and depths one/two are measured separately. These "
        "incremental gains depend on the preceding optimizations; isolated operator "
        "speedups do not predict end-to-end speedups.",
        "",
        "The operator graph captures the GDN boundary alone; the service graph captures "
        "the full decode model. Their speedup ratios therefore measure different scopes.",
        "",
        "Operator CUDA-event timing excludes initial-state restoration. Service "
        "requests use identical "
        "engineering-review, code and Chinese test-planning prompts, fixed lengths, "
        "temperature zero, disabled prefix caching "
        "and closed-loop concurrency. "
        "Throughput counts emitted tokens. TPOT averages the interval from first to "
        "last emitted token; "
        "MTP may deliver multiple tokens in one SSE event. Warmup uses the actual "
        "concurrent workload and is excluded from timing.",
        "",
        "Generation checks compare fixed English, Chinese and code prompts to native "
        "vLLM token sequences. "
        "They are regression checks, not a general model-quality benchmark. Peak device "
        "memory includes weights, "
        "reserved KV cache, graph allocations and process overhead, sampled every 200 ms.",
        "The runtime reserves cache against the same 90% device-memory budget, so total "
        "device memory is not a direct measure of transient state-copy savings.",
        "",
        "The SM90 decoder uses 55 registers/thread, 46,144 bytes of shared memory and "
        "zero local-memory bytes according to "
        "[cuobjdump resource output](cuda-resources.txt). Separate operator profiler "
        "event summaries in `results/profiles/` confirm that A and B execute "
        "`index_select` and `index_copy_`, while C "
        "executes the native decode without these state copies. Event timing follows "
        "an untimed state restore and may benefit from a warm cache; it is not an HBM "
        "bandwidth or achieved-occupancy measurement.",
        "",
        "Precision and state semantics are documented in [gdn-contract.md](gdn-contract.md).",
    ]
    (args.output / "performance.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
