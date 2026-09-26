from types import SimpleNamespace

import torch
from lightdelta.runtime.speculation import LightDeltaRejectionSampler


def test_greedy_mtp_prefix_verification():
    batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens_per_req=[2, 2],
        input_ids=torch.tensor([10, 11, 20, 21]),
        query_start_loc=torch.tensor([0, 2, 4]),
        cu_num_logits=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4]),
        prefill_len_np=[0, 0],
    )
    logits = torch.full((6, 40), -100.0)
    for row, token in enumerate([10, 11, 12, 20, 31, 32]):
        logits[row, token] = 1
    output = LightDeltaRejectionSampler(2)(logits, batch)
    assert output.sampled_token_ids.tolist() == [[10, 11, 12], [20, 31, -1]]
    assert output.num_sampled.tolist() == [3, 2]
    assert output.num_rejected.tolist() == [0, 1]
