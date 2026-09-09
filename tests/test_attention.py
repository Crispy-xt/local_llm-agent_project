import torch

from tiny_lm.attention import CausalSelfAttention


def test_attention_output_shape() -> None:
    attention = CausalSelfAttention(n_embd=8, n_head=2, block_size=16)
    output = attention(torch.randn(3, 5, 8))
    assert output.shape == (3, 5, 8)


def test_future_tokens_do_not_change_previous_output() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(n_embd=8, n_head=2, block_size=16)
    attention.eval()
    prefix = torch.randn(1, 3, 8)
    first = attention(prefix)
    extended = torch.cat((prefix, torch.randn(1, 2, 8)), dim=1)
    second = attention(extended)
    assert torch.allclose(first, second[:, :3], atol=1e-6)