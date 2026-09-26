"""Read the local model geometry without loading weights or importing vLLM."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GDNGeometry:
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    gdn_layers: int
    attention_layers: int
    norm_eps: float

    @classmethod
    def from_model(cls, directory: str | Path) -> "GDNGeometry":
        config = json.loads((Path(directory) / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            raise ValueError("Expected a Qwen3.5-compatible Qwen3.8 checkpoint")
        text = config["text_config"]
        result = cls(
            key_heads=text["linear_num_key_heads"],
            value_heads=text["linear_num_value_heads"],
            key_dim=text["linear_key_head_dim"],
            value_dim=text["linear_value_head_dim"],
            gdn_layers=text["layer_types"].count("linear_attention"),
            attention_layers=text["layer_types"].count("full_attention"),
            norm_eps=text["rms_norm_eps"],
        )
        if min(result.key_heads, result.value_heads, result.key_dim, result.value_dim) <= 0:
            raise ValueError("Head counts and dimensions must be positive")
        if result.value_heads % result.key_heads:
            raise ValueError("Value heads must be divisible by key heads")
        return result

    def as_dict(self) -> dict:
        return asdict(self)
