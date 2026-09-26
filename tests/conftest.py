import pytest
import torch


@pytest.fixture
def cuda_case():
    if not torch.cuda.is_available():
        pytest.skip("run CUDA tests through mlx worker login")

    def make(batch=4, width=1):
        torch.manual_seed(42)
        device = "cuda"
        n = batch * width
        # Strided rows exercise the actual packed qkvz and ba projection layouts.
        qkvz = torch.randn(n, 16384, device=device, dtype=torch.bfloat16) * 0.2
        ba = torch.randn(n, 96, device=device, dtype=torch.bfloat16)
        slots = torch.arange(1, n + 1, device=device, dtype=torch.int32).view(batch, width)
        return dict(
            qkv=qkvz[:, :10240],
            a=ba[:, 48:],
            b=ba[:, :48],
            a_log=torch.randn(48, device=device),
            dt_bias=torch.randn(48, device=device, dtype=torch.bfloat16),
            slots=slots,
            cu=torch.arange(0, n + 1, width, device=device, dtype=torch.int32),
            accepted=torch.ones(batch, device=device, dtype=torch.int32),
            pool=torch.randn(n + 2, 48, 128, 128, device=device) * 0.1,
            gate=qkvz[:, 10240:].view(n, 48, 128),
            weight=torch.randn(128, device=device, dtype=torch.bfloat16),
            out=torch.empty(n, 48, 128, device=device, dtype=torch.bfloat16),
        )

    return make
