import torch


class CasualAttention(torch.nn.Module):
    """ Casual Attentionクラス

    Args:
        d_in (int):     入力次元数
        d_out (int):    出力次元数
        context_length (int):   トークン数（マスクサイズ）
        dropout (float):    ドロップアウト確率
        is_bias (bool):     bias項使用可否

    Attributes:
        W_query (torch.nn.Parameter):   訓練可能なクエリベクトル
        W_key (torch.nn.Parameter):     訓練可能なキーベクトル
        W_value (torch.nn.Parameter):   訓練可能な値ベクトル
        dropout (torch.nn.Dropout):     ドロップアウト層
        register_buffer:  self.mask 上三角形行列

    Notes:
        ・register_buffer()を使用することで、テンソルを適切なデバイス（CPU, GPU）に自動的に移動してくれる。
        ・マスクをスライスすることで、マスクサイズが入力トークン数よりも大きい場合に備えている。
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout:float, is_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values

        return context_vec


if __name__ == "__main__":

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch)
    d_in = inputs.shape[1]
    d_out = 2

    casual_attention = CasualAttention(d_in, d_out, batch.shape[1], 0.0)
    context_vecs = casual_attention(batch)
    print(context_vecs)

