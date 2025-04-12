import torch
from chapter3.self_attention_class import SelfAttentionV2


def create_simple_mask(inputs, d_in, d_out):
    # Self-Attentionクラスのクエリとキーの重みを再利用
    self_attention2 = SelfAttentionV2(d_in, d_out)
    queries = self_attention2.W_query(inputs)
    keys = self_attention2.W_key(inputs)

    attention_scores = queries @ keys.T
    attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
    print("attentionの重み：")
    print(attention_weights)

    # マスク生成（下三角形の行列）
    context_length = attention_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print("mask:")
    print(mask_simple)

    # マスク適用
    masked_simple = attention_weights * mask_simple
    print("mask適用結果:")
    print(masked_simple)

    # 正規化
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)

    return masked_simple_norm


def create_mask(inputs, d_in, d_out):
    # Self-Attentionクラスのクエリとキーの重みを再利用
    self_attention2 = SelfAttentionV2(d_in, d_out)
    queries = self_attention2.W_query(inputs)
    keys = self_attention2.W_key(inputs)

    attention_scores = queries @ keys.T
    attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
    print("attentionの重み：")
    print(attention_weights)

    # マスク生成（上三角形行列）
    context_length = inputs.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attention_scores.masked_fill(mask.bool(), -torch.inf)  # 1の部分を-infinityに変換
    print("マスク：")
    print(masked)

    # 正規化
    attention_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
    print("attentionの重み")
    print(attention_weights)

    return attention_weights


# トークンIDの埋め込みベクトル
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

d_in = inputs.shape[1]  # 入力次元: 3
d_out = 2  # 出力次元

torch.manual_seed(789)

# attention_weights = create_simple_mask(inputs, d_in, d_out)
attention_weights = create_mask(inputs, d_in, d_out)

torch.manual_seed(123)

# ドロップアウト適用
dropout = torch.nn.Dropout(0.5)
print("ドロップアウト適用後のattentionの重み")
print(dropout(attention_weights))