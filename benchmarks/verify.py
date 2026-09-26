"""Generation regression check used by the unified experiment runner."""

import json
import os
from pathlib import Path

import httpx
from transformers import AutoTokenizer

PROMPTS = (
    "What is 17 + 25? Answer with only the number.",
    "用一句话解释什么是 CUDA。",
    "Write a Python function that adds two numbers.",
)


def _tokens(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


async def check(url: str, output: Path, reference: Path | None = None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(os.environ["LIGHTDELTA_MODEL_PATH"])
    records = []
    async with httpx.AsyncClient(base_url=url, timeout=600) as client:
        for prompt in PROMPTS:
            chunks, output_ids = [], []
            async with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "lightdelta",
                    "prompt": _tokens(tokenizer, prompt),
                    "max_tokens": 64,
                    "temperature": 0,
                    "seed": 0,
                    "stream": True,
                    "return_token_ids": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        event = json.loads(line[6:])
                        for choice in event.get("choices", []):
                            chunks.append(choice.get("text", ""))
                            output_ids.extend(choice.get("token_ids") or [])
            records.append({"prompt": prompt, "text": "".join(chunks), "token_ids": output_ids})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    if reference is not None:
        expected = json.loads(reference.read_text())
        for actual, baseline in zip(records, expected, strict=True):
            if actual["token_ids"] != baseline["token_ids"]:
                raise AssertionError(f"generation differs from native control: {actual['prompt']}")
