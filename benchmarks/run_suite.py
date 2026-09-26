"""Restart the same runtime for each immutable ablation configuration."""

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from service import measure
from transformers import AutoTokenizer
from verify import check
from workload import prompts


async def wait_ready(process, url):
    deadline = time.monotonic() + 1200
    async with httpx.AsyncClient(base_url=url, timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited with status {process.returncode}; inspect server.log"
                )
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Server did not become ready within 20 minutes")


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "reference",
            "fusion",
            "indexed",
            "graph",
            "full",
            "full_no_graph",
            "full_packed",
            "full_prefill",
            "mtp1",
            "mtp2",
        ],
    )
    parser.add_argument("--inputs", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--outputs", nargs="+", type=int, default=[128])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--request-waves", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/service"))
    parser.add_argument("--reference", type=Path, default=Path("results/native/control.json"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(os.environ["LIGHTDELTA_MODEL_PATH"])
    prompt_sets = {length: prompts(tokenizer, length) for length in args.inputs}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prompts.json").write_text(json.dumps(prompt_sets))
    for repeat in range(args.rounds):
        # Reverse the order on alternating rounds to limit warm-cache/time-order bias.
        order = args.configs if repeat % 2 == 0 else list(reversed(args.configs))
        for name in order:
            folder = args.output / f"round-{repeat}" / name
            folder.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, LIGHTDELTA_PROBE_DIR=str(folder.resolve()), PYTHONUNBUFFERED="1")
            with (folder / "server.log").open("w") as log, (folder / "memory.csv").open("w") as mem:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "lightdelta.cli",
                        "serve",
                        "--config",
                        f"configs/{name}.toml",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "--loop-ms=200",
                    ],
                    stdout=mem,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    await wait_ready(server, "http://127.0.0.1:8000")
                    print(f"Ready: round={repeat} config={name}", flush=True)
                    await check(
                        "http://127.0.0.1:8000",
                        folder / "control.json",
                        args.reference,
                    )
                    for inputs in args.inputs:
                        for outputs in args.outputs:
                            for concurrency in args.concurrency:
                                result = await measure(
                                    "http://127.0.0.1:8000",
                                    prompt_sets[inputs],
                                    outputs,
                                    concurrency,
                                    math.ceil(concurrency * args.request_waves / 3) * 3,
                                )
                                result.update(
                                    config=name, round=repeat, workload="engineering-code-chinese"
                                )
                                target = folder / f"i{inputs}-o{outputs}-c{concurrency}.json"
                                target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                                print(
                                    f"{target}: {result['output_tokens_per_s']:.2f} tok/s",
                                    flush=True,
                                )
                finally:
                    stop(server)
                    stop(monitor)


if __name__ == "__main__":
    asyncio.run(main())
