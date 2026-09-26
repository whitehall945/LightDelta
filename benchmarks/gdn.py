"""State restoration is outside every CUDA-event timing interval."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from lightdelta.backend import GDNBackend
from lightdelta.config import BackendConfig


def run(batch, width, config, iterations):
    torch.manual_seed(42)
    n = batch * width
    qkvz = torch.randn(n, 16384, device="cuda", dtype=torch.bfloat16) * 0.2
    ba = torch.randn(n, 96, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(n + 1, 48, 128, 128, device="cuda") * 0.1
    state = pool.clone()
    case = dict(
        qkv=qkvz[:, :10240],
        a=ba[:, 48:],
        b=ba[:, :48],
        a_log=torch.randn(48, device="cuda"),
        dt_bias=torch.randn(48, device="cuda", dtype=torch.bfloat16),
        slots=torch.arange(1, n + 1, device="cuda", dtype=torch.int32).view(batch, width),
        cu=torch.arange(0, n + 1, width, device="cuda", dtype=torch.int32),
        accepted=torch.ones(batch, device="cuda", dtype=torch.int32),
        pool=pool,
        gate=qkvz[:, 10240:].view(n, 48, 128),
        weight=torch.randn(128, device="cuda", dtype=torch.bfloat16),
        out=torch.empty(n, 48, 128, device="cuda", dtype=torch.bfloat16),
    )
    backend = GDNBackend(config)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            pool.copy_(state)
            backend(**case)
    torch.cuda.current_stream().wait_stream(stream)
    if config.cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend(**case)
        execute = graph.replay
    else:

        def execute():
            backend(**case)

    timers = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        pool.copy_(state)
        start.record()
        execute()
        end.record()
        timers.append((start, end))
    torch.cuda.synchronize()
    values = [a.elapsed_time(b) * 1000 for a, b in timers]
    return {
        "batch": batch,
        "tokens_per_request": width,
        "median_us": statistics.median(values),
        "samples_us": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/operator.json"))
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    results = []
    for batch in (1, 4, 8):
        for width in (1, 2, 3, 4):
            for name in ("reference", "fusion", "indexed", "graph"):
                config = BackendConfig.from_file(f"configs/{name}.toml")
                record = {"config": name, **run(batch, width, config, args.iterations)}
                results.append(record)
                print(f"{name:10s} B={batch} T={width}: {record['median_us']:.2f} us", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "measurements": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
