"""Deterministic natural-language workloads with exact prompt-token lengths."""

CASES = [
    (
        "A request enters a queue, is assigned a state slot, runs a prefill pass, and then "
        "produces tokens in decode steps. Finished requests release their slots. Requests may "
        "have different prompt lengths and output limits. CUDA graphs reuse fixed buffers. "
        "The scheduler must isolate state across users and avoid reading padding slots. "
        "Measurements record time to first token, time per output token and total throughput. ",
        "Write a detailed engineering review of this inference service. Cover correctness, "
        "memory ownership, batching, testing and measurement. Explain concrete failure cases "
        "and remedies, with at least eight substantial numbered points.",
    ),
    (
        "We need a bounded least-recently-used cache. Keys are strings and values can be any "
        "Python object. Reading or updating an entry makes it most recently used. Inserting "
        "a new key when the cache is full evicts the least recently used entry. Capacity is "
        "positive. Missing keys raise KeyError. Updating an existing key does not grow the cache. "
        "The implementation is used by one thread and should provide constant-time operations. ",
        "Implement this cache in Python. Provide complete typed code, docstrings, a short "
        "usage example, and at least six meaningful unit tests for its edge cases. Explain "
        "the time and space complexity after the code.",
    ),
    (
        "项目有一个共享状态池，每个请求使用独立的槽位。长输入先进行预填充，随后逐步生成文本。"
        "投机解码一次提出多个候选，主模型验证后只保留已接受前缀对应的状态。"
        "请求结束后槽位可以复用，但不能泄漏旧请求的数据。图执行需要稳定的内存地址。"
        "性能实验必须固定模型、输入、输出长度和并发数，分别测量延迟、吞吐和显存。",
        "请给这个项目写一份详细的测试方案，包含数学正确性、状态恢复、混合批次、"
        "槽位复用、图重放、流式接口和性能消融。每一项都给出具体输入、验证方法和失败条件，"
        "最后给出实施顺序。",
    ),
]


def prompts(tokenizer, input_tokens):
    result = []
    marker = "[[LIGHTDELTA_CONTEXT]]"
    for context, task in CASES:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Context:\n{marker}\n\nTask:\n{task}"}],
            tokenize=False,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode(after, add_special_tokens=False)
        budget = input_tokens - len(prefix) - len(suffix)
        if budget <= 0:
            raise ValueError("Prompt length must fit the chat template and task instructions")
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        body = (context_ids * ((budget + len(context_ids) - 1) // len(context_ids)))[:budget]
        result.append(prefix + body + suffix)
    return result
