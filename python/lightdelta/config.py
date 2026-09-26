"""Immutable process-level experiment configuration; fail closed on missing paths."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendConfig:
    compute: str = "torch_reference"
    state_io: str = "packed"
    prefill_io: str = "reference"
    torch_compile: bool = False
    cuda_graph: bool = False
    speculation_method: str = "none"
    num_speculative_tokens: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "BackendConfig":
        data = tomllib.loads(Path(path).read_text())
        allowed = {
            "gdn": {"compute", "state_io", "prefill_io"},
            "runtime": {"torch_compile", "cuda_graph"},
            "speculation": {"method", "num_speculative_tokens"},
        }
        if data.keys() - allowed.keys():
            raise ValueError(f"Unknown config sections: {data.keys() - allowed.keys()}")
        for section, values in data.items():
            if not isinstance(values, dict) or values.keys() - allowed[section]:
                raise ValueError(f"Unknown keys or invalid table in [{section}]")
        spec = data.get("speculation", {})
        result = cls(
            **data.get("gdn", {}),
            **data.get("runtime", {}),
            speculation_method=spec.get("method", "none"),
            num_speculative_tokens=spec.get("num_speculative_tokens", 0),
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected = BackendConfig()
        for field in self.__dataclass_fields__:
            if type(getattr(self, field)) is not type(getattr(expected, field)):
                raise ValueError(f"Invalid type for {field}")
        if self.compute not in ("torch_reference", "cuda_fused"):
            raise ValueError("compute must be torch_reference or cuda_fused")
        if self.state_io not in ("packed", "indexed"):
            raise ValueError("state_io must be packed or indexed")
        if self.prefill_io not in ("reference", "fused_copy"):
            raise ValueError("prefill_io must be reference or fused_copy")
        if self.torch_compile:
            raise ValueError("torch.compile is disabled for controlled ablations")
        if self.compute == "torch_reference" and (self.state_io != "packed" or self.cuda_graph):
            raise ValueError("Torch reference requires packed state and eager execution")
        if self.speculation_method == "none":
            if self.num_speculative_tokens != 0:
                raise ValueError("none speculation requires zero draft tokens")
        elif self.speculation_method != "mtp" or self.num_speculative_tokens not in (1, 2, 3):
            raise ValueError("MTP requires 1, 2 or 3 draft tokens")
