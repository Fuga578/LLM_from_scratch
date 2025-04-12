import torch
from chapter3.casual_attention import CasualAttention


class MultiHeadAttentionWrapper(torch.nn.Module):
    """ Casual Attentionクラス

    Args:
        d_in (int):     入力次元数
        d_out (int):    出力次元数
        context_length (int):   トークン数（マスクサイズ）
        dropout (float):    ドロップアウト確率
        num_heads (int):    ヘッド数
        is_bias (bool):     bias項使用可否
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, is_bias=False):
        super().__init__()

        self.heads = torch.nn.ModuleList(
            [CasualAttention(d_in, d_out, context_length, dropout, is_bias) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]]
)
batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1]
d_in = batch.shape[2]
d_out = 2

multi_head_attention = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vecs = multi_head_attention(batch)

print(context_vecs)
