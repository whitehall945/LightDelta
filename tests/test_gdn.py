import pytest
import torch
from lightdelta.config import BackendConfig
from lightdelta.reference import GDNInputs, post_conv_gdn


def test_reference_prefix_states():
    torch.manual_seed(7)
    batch, tokens, hk, hv, k, v = 3, 4, 2, 6, 5, 7
    rand = lambda *shape: torch.randn(shape)  # noqa: E731
    x = GDNInputs(
        rand(batch, tokens, hk, k),
        rand(batch, tokens, hk, k),
        rand(batch, tokens, hv, v),
        rand(batch, tokens, hv),
        rand(batch, tokens, hv),
        rand(hv),
        rand(hv),
        rand(batch, tokens, hv, v),
        rand(v),
    )
    state = rand(batch, hv, v, k)
    original = state.clone()
    lengths = torch.tensor([4, 2, 0])
    full = post_conv_gdn(x, state, lengths, save_checkpoints=True)
    running = state
    for t in range(tokens):
        step = GDNInputs(
            **{
                name: (
                    getattr(x, name)[:, t : t + 1]
                    if name in ("q", "k", "v", "a", "b", "gate")
                    else getattr(x, name)
                )
                for name in x.__dataclass_fields__
            }
        )
        result = post_conv_gdn(step, running, (lengths > t).int())
        torch.testing.assert_close(result.output[:, 0], full.output[:, t])
        torch.testing.assert_close(result.final_state, full.checkpoints[:, t + 1])
        running = result.final_state
    torch.testing.assert_close(state, original, rtol=0, atol=0)
    torch.testing.assert_close(full.final_state[2], state[2], rtol=0, atol=0)
    assert full.output[2].count_nonzero() == 0


def test_scalar_recurrence():
    # One-dimensional analytic recurrence, including decay before the delta update.
    one = torch.ones(1, 1, 1, 1)
    zero = torch.zeros(1, 1, 1)
    x = GDNInputs(one, one, one * 3, zero, zero, torch.zeros(1), torch.zeros(1), one, torch.ones(1))
    result = post_conv_gdn(x, one * 2, torch.ones(1, dtype=torch.int32))
    key = (1 + 1e-6) ** -0.5
    expected = 1 + 0.5 * (3 - key) * key
    torch.testing.assert_close(result.final_state, one * expected)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("state_io", ["packed", "indexed"])
@pytest.mark.cuda
def test_cuda_matches_reference(cuda_case, width, state_io):
    from lightdelta.backend import GDNBackend

    case = cuda_case(width=width)
    case["accepted"][:] = torch.arange(4, device="cuda") % width + 1
    # Reserve a padding request and a discarded candidate boundary.
    case["slots"][-1] = 0
    before = case["pool"].clone()
    GDNBackend(BackendConfig())(**case)
    expected_output, expected_state = case["out"].clone(), case["pool"].clone()
    case["pool"].copy_(before)
    GDNBackend(BackendConfig(compute="cuda_fused", state_io=state_io))(**case)
    torch.testing.assert_close(case["out"], expected_output, atol=0.008, rtol=0.02)
    torch.testing.assert_close(case["pool"], expected_state, atol=3e-6, rtol=3e-4)
    torch.testing.assert_close(case["pool"][0], before[0], atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("state_io", ["packed", "indexed"])
def test_graph_replay_changed_state_indices(cuda_case, state_io):
    from lightdelta.backend import GDNBackend

    case = cuda_case(batch=2, width=3)
    backend = GDNBackend(BackendConfig(compute="cuda_fused", state_io=state_io))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            backend(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend(**case)
    case["slots"].copy_(case["slots"].flip(0))
    case["accepted"].fill_(2)
    before = case["pool"].clone()
    GDNBackend(BackendConfig())(**case)
    expected_out, expected_state = case["out"].clone(), case["pool"].clone()
    case["pool"].copy_(before)
    graph.replay()
    torch.testing.assert_close(case["out"], expected_out, atol=0.008, rtol=0.02)
    torch.testing.assert_close(case["pool"], expected_state, atol=3e-6, rtol=3e-4)


@pytest.mark.cuda
@pytest.mark.parametrize("state_io", ["packed", "indexed"])
def test_ragged_verification_and_discarded_slots(cuda_case, state_io):
    from lightdelta.backend import GDNBackend

    case = cuda_case(width=4)
    case["cu"].copy_(torch.tensor([0, 4, 7, 8, 8], device="cuda", dtype=torch.int32))
    case["accepted"][0] = 0
    case["slots"][1, 2] = 0
    for name in ("qkv", "a", "b", "gate", "out"):
        case[name] = case[name][:8]
    initial = case["pool"].clone()
    GDNBackend(BackendConfig())(**case)
    expected = case["out"].clone(), case["pool"].clone()
    case["pool"].copy_(initial)
    GDNBackend(BackendConfig(compute="cuda_fused", state_io=state_io))(**case)
    torch.testing.assert_close(case["out"], expected[0], atol=0.008, rtol=0.02)
    torch.testing.assert_close(case["pool"], expected[1], atol=3e-6, rtol=3e-4)


@pytest.mark.cuda
@pytest.mark.parametrize("index_stride", [1, 4])
def test_prefill_state_initialization_and_reuse(cuda_case, index_stride):
    from lightdelta import ops  # noqa: F401

    case = cuda_case(batch=2)
    pool = case["pool"]
    before = pool.clone()
    slots = torch.full((3, index_stride), -1, device="cuda", dtype=torch.int32)[:, 0]
    slots.copy_(torch.tensor([2, 1, 0], device="cuda", dtype=torch.int32))
    valid = torch.tensor([True, False, False], device="cuda")
    packed = torch.empty(3, *pool.shape[1:], device="cuda")
    torch.ops.lightdelta.gather_state(pool, slots, valid, packed)
    torch.testing.assert_close(packed[0], pool[2], rtol=0, atol=0)
    assert packed[1:].count_nonzero() == 0
    packed.add_(1)
    torch.ops.lightdelta.scatter_state(packed, slots, pool)
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[1], torch.ones_like(pool[1]), rtol=0, atol=0)
