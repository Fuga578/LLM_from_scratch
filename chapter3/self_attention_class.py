import torch


class SelfAttentionV1(torch.nn.Module):
    """ 訓練可能な重みを持つSelf-Attentionクラス（Scaled Dot-Product Attention）

    Args:
        d_in (int):     入力次元数
        d_out (int):    出力次元数

    Attributes:
        W_query (torch.nn.Parameter):   訓練可能なクエリベクトル
        W_key (torch.nn.Parameter):     訓練可能なキーベクトル
        W_value (torch.nn.Parameter):   訓練可能な値ベクトル

    Notes:
        attentionスコアを埋め込み次元サイズで正規化しているのは、勾配消失を防ぐため
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x) -> torch.Tensor:
        """ 入力に対してコンテキストベクトルを算出します。

        Args:
            x (torch.Tensor):   入力トークンベクトル

        Returns:
            torch.Tensor:   コンテキストベクトル
        """

        # 入力トークンベクトルを射影
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        # アテンションスコア（クエリベクトル * キーベクトル）
        attention_scores = queries @ keys.T

        # 正規化
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        # コンテキストベクトル（attentionの重み * 値ベクトル）
        context_vec = attention_weights @ values

        return context_vec

class SelfAttentionV2(torch.nn.Module):
    """ 訓練可能な重みを持つSelf-Attentionクラス（Scaled Dot-Product Attention）

    torch.nn.Linearを使用。
    bias項がなければ、実質的に行列積を計算する。

    Args:
        d_in (int):     入力次元数
        d_out (int):    出力次元数
        is_bias (bool): bias項使用可否

    Attributes:
        W_query (torch.nn.Linear):   訓練可能なクエリベクトル
        W_key (torch.nn.Linear):     訓練可能なキーベクトル
        W_value (torch.nn.Linear):   訓練可能な値ベクトル

    Notes:
        attentionスコアを埋め込み次元サイズで正規化しているのは、勾配消失を防ぐため
    """

    def __init__(self, d_in: int, d_out: int, is_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 入力に対してコンテキストベクトルを算出します。

        Args:
            x (torch.Tensor):   入力トークンベクトル

        Returns:
            torch.Tensor:   コンテキストベクトル
        """

        # 入力トークンベクトルを射影
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # アテンションスコア（クエリベクトル * キーベクトル）
        attention_scores = queries @ keys.T

        # 正規化
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        # コンテキストベクトル（attentionの重み * 値ベクトル）
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
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)

    self_attention1 = SelfAttentionV1(d_in, d_out)
    self_attention2 = SelfAttentionV2(d_in, d_out)
    print(f"ver1: {self_attention1(inputs)}")
    print(f"ver2: {self_attention2(inputs)}")
