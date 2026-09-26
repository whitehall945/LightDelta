import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lightdelta.config import BackendConfig
from lightdelta.model import GDNGeometry


def serve(args):
    config = BackendConfig.from_file(args.config)
    geometry = GDNGeometry.from_model(args.model)
    if (
        geometry.key_heads,
        geometry.value_heads,
        geometry.key_dim,
        geometry.value_dim,
        geometry.gdn_layers,
        geometry.attention_layers,
    ) != (16, 48, 128, 128, 48, 16):
        raise ValueError("The service requires the Qwen3.8-27B model geometry")
    os.environ["LIGHTDELTA_CONFIG"] = str(Path(args.config).resolve())
    os.environ["VLLM_GDN_DECODE_KERNEL"] = "cuda"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    compilation = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY" if config.cuda_graph else "NONE"}
    if config.cuda_graph:
        width = config.num_speculative_tokens + 1
        compilation["cudagraph_capture_sizes"] = [b * width for b in (1, 2, 4, 8)]
    command = [
        shutil.which("vllm"),
        "serve",
        args.model,
        "--served-model-name",
        "lightdelta",
        "--host",
        args.host,
        "--port",
        str(args.port),
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
        "--compilation-config",
        json.dumps(compilation),
        "--additional-config",
        '{"gdn_prefill_backend":"flashinfer"}',
        "--seed",
        "0",
        "--worker-cls",
        "lightdelta.runtime.worker.LightDeltaWorker",
    ]
    if config.speculation_method == "mtp":
        command += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": config.num_speculative_tokens}),
            "--per-request-spec-decode-metrics",
            "summary",
        ]
    if not config.cuda_graph:
        command.append("--enforce-eager")
    os.execv(command[0], command)


def main():
    parser = argparse.ArgumentParser(prog="lightdelta")
    commands = parser.add_subparsers(dest="command", required=True)
    model_path = os.environ.get(
        "LIGHTDELTA_MODEL_PATH", "/mnt/bn/search-nlp-us/wanghenglong/Qwen3.8-27B"
    )
    inspect = commands.add_parser("inspect-model")
    inspect.add_argument("--model", default=model_path)
    server = commands.add_parser("serve")
    server.add_argument("--config", required=True)
    server.add_argument("--model", default=model_path)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    experiment = commands.add_parser("experiment")
    experiment.add_argument("--rounds", type=int, default=1)
    experiment.add_argument("--results", type=Path, default=Path("results"))
    experiment.add_argument("--report-output", type=Path, default=Path("docs"))
    experiment.add_argument("--skip-operator", action="store_true")
    experiment.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect-model":
        print(json.dumps(GDNGeometry.from_model(args.model).as_dict(), indent=2))
    elif args.command == "experiment":
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(root / "benchmarks/run.py"),
            "--rounds",
            str(args.rounds),
            "--results",
            str(args.results),
            "--report-output",
            str(args.report_output),
        ]
        if args.skip_operator:
            command.append("--skip-operator")
        if args.skip_native:
            command.append("--skip-native")
        subprocess.run(command, cwd=root, check=True)
    else:
        serve(args)


if __name__ == "__main__":
    main()
