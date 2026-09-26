"""Single entry point for the final LightDelta experiment."""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command, env=None):
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Result root for this run",
    )
    parser.add_argument("--report-output", type=Path, default=Path("docs"))
    parser.add_argument("--skip-operator", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    native_control = args.results / "native" / "control.json"
    if not args.skip_native:
        env = dict(os.environ)
        env.pop("LIGHTDELTA_CONFIG", None)
        env.pop("VLLM_GDN_DECODE_KERNEL", None)
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        command = [
            "vllm",
            "serve",
            env["LIGHTDELTA_MODEL_PATH"],
            "--served-model-name",
            "lightdelta",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--language-model-only",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "4096",
            "--gpu-memory-utilization",
            "0.90",
            "--mamba-ssm-cache-dtype",
            "float32",
            "--no-enable-prefix-caching",
            "--enforce-eager",
            "--additional-config",
            '{"gdn_prefill_backend":"flashinfer"}',
        ]
        server = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        try:
            import time

            import httpx

            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"native vLLM exited with {server.returncode}")
                try:
                    if httpx.get("http://127.0.0.1:8000/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("native vLLM did not become ready")
            import asyncio

            from verify import check

            native_control.parent.mkdir(parents=True, exist_ok=True)
            asyncio.run(check("http://127.0.0.1:8000", native_control))
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            server.wait(timeout=60)
    if not args.skip_operator:
        _run([sys.executable, "benchmarks/gdn.py", "--output", str(args.results / "operator.json")])
    _run(
        [
            sys.executable,
            "benchmarks/run_suite.py",
            "--rounds",
            str(args.rounds),
            "--output",
            str(args.results / "service"),
            "--reference",
            str(native_control),
        ]
    )
    _run(
        [
            sys.executable,
            "benchmarks/report.py",
            "--results",
            str(args.results),
            "--output",
            str(args.report_output),
        ]
    )


if __name__ == "__main__":
    main()
