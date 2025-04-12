import torch


class MultiHeadAttention(torch.nn.Module):
    """ Multi-head Attentionクラス

    Args:
        d_in (int): 入力次元数
        d_out (int): 出力次元数
        context_length (int):   トークン長さ（= マスクサイズ）
        dropout (float):    ドロップアウト確率
        num_heads (int):    attentionヘッド数（並列数）
        is_bias (bool):     バイアス項使用可否

    Notes:
        ・各ヘッドの出力次元数を均等にするために、トータルの出力次元数（d_out）がヘッド数（num_heads）で割り切れないならassert。
        ・self.out_projについて、厳密には必要ないが、多くのLLMアーキテクチャで使用されているため、一応追加している。
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, is_bias: bool=False):
        super().__init__()

        # 出力次元数チェック
        assert (d_out % num_heads == 0), "出力次元数（d_in）はヘッド数（num_heads）で割り切れる値に設定してください。"

        self.d_out = d_out
        self.num_heads = num_heads

        # 各headの次元数
        self.head_dim = d_out // num_heads

        # Q, K, Vの重み行列（線形変換）
        self.W_query = torch.nn.Linear(d_in, d_out, bias=is_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=is_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=is_bias)

        # コンテキストベクトル全体を線形変換する用の行列
        self.out_proj = torch.nn.Linear(d_out, d_out)

        # ドロップアウト層
        self.dropout = torch.nn.Dropout(dropout)

        # マスク（上三角形行列）作成
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): (batch_size, num_tokens, d_in) 次元のデータ

        Returns:
            torch.Tensor:   コンテキストベクトル (batch_size, num_tokens, d_out) 次元のデータ
        """

        # 次元数取得
        batch_size, num_tokens, d_in = x.shape

        # Q, K, V行列生成
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 行列をhead数に分割
        # (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim）
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # 次元並び変え（並列に計算したいため、head数を2番目に）
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # attentionスコア算出
        # quries: (batch_size, num_heads, num_tokens, head_dim)
        # keys.T: (batch_size, num_heads, head_dim, num_tokens)
        # -> (batch_size, num_heads, num_tokens, num_tokens)
        attention_scores = queries @ keys.transpose(2, 3)

        # マスク整形
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # マスク処理適用
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # 正規化
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # コンテキストベクトル算出 (batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attention_weights @ values).transpose(1, 2)

        # 並列のコンテキストベクトルを1つに結合
        # ※ contiguoutにより演算速度向上
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        # コンテキストベクトルを再整理
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
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

    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
