"""Measure SSE TTFT/TPOT and retain raw responses; run on the GPU worker."""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer
from workload import prompts


async def completion(client, prompt, output_tokens, semaphore):
    async with semaphore:
        start = time.perf_counter()
        arrivals, chunks, token_ids, usage, metrics = [], [], [], None, None
        async with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "lightdelta",
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0,
                "seed": 0,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("metrics"):
                    metrics = event["metrics"]
                for choice in event.get("choices", []):
                    if choice.get("token_ids"):
                        arrivals.append(time.perf_counter() - start)
                        token_ids.extend(choice["token_ids"])
                        chunks.append(choice["text"])
        if usage is None or not arrivals:
            raise RuntimeError("Stream did not contain output and final token usage")
        count = usage["completion_tokens"]
        return {
            "ttft_s": arrivals[0],
            "latency_s": arrivals[-1],
            "tpot_s": (arrivals[-1] - arrivals[0]) / max(count - 1, 1),
            "output_tokens": count,
            "input_tokens": usage["prompt_tokens"],
            "arrival_s": arrivals,
            "token_ids": token_ids,
            "metrics": metrics,
            "text": "".join(chunks),
        }


async def measure(base_url, prompt_set, output_tokens, concurrency, requests):
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        semaphore = asyncio.Semaphore(concurrency)
        await asyncio.gather(
            *[
                completion(
                    client, prompt_set[i % len(prompt_set)], min(output_tokens, 16), semaphore
                )
                for i in range(requests)
            ]
        )
        before = (await client.get("/metrics")).text
        start = time.perf_counter()
        records = await asyncio.gather(
            *[
                completion(client, prompt_set[i % len(prompt_set)], output_tokens, semaphore)
                for i in range(requests)
            ]
        )
        elapsed = time.perf_counter() - start
        after = (await client.get("/metrics")).text
    return {
        "concurrency": concurrency,
        "requests": requests,
        "elapsed_s": elapsed,
        "output_tokens_per_s": sum(r["output_tokens"] for r in records) / elapsed,
        "median_ttft_s": statistics.median(r["ttft_s"] for r in records),
        "median_tpot_s": statistics.median(r["tpot_s"] for r in records),
        "records": records,
        "metrics_before": before,
        "metrics_after": after,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(os.environ["LIGHTDELTA_MODEL_PATH"])
    prompt_set = prompts(tokenizer, args.input_tokens)
    result = asyncio.run(
        measure(args.url, prompt_set, args.output_tokens, args.concurrency, args.requests)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("records", "metrics_before", "metrics_after")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
