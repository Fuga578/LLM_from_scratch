import torch


# トークンIDの埋め込みベクトル
inputs = torch.tensor(
 [[0.43, 0.15, 0.89],  # Your     (x^1)
  [0.55, 0.87, 0.66],  # journey  (x^2)
  [0.57, 0.85, 0.64],  # starts   (x^3)
  [0.22, 0.58, 0.33],  # with     (x^4)
  [0.77, 0.25, 0.10],  # one      (x^5)
  [0.05, 0.80, 0.55]]  # step     (x^6)
)

x_2 = inputs[1]     # 入力（2番目のトークン）
d_in = inputs.shape[1]  # 入力次元: 3
d_out = 2   # 出力次元

torch.manual_seed(123)

# 訓練可能な重み行列
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 入力トークンからクエリベクトルなどを作成
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value

# クエリベクトル、キーベクトルから、attentionスコア算出
attention_scores_2 = query_2 @ keys.T
print(f"attentionスコア2: {attention_scores_2}")

# ソフトマックス関数で正規化（スケールした値を正規化）
d_k = keys.shape[-1]
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)
print(f"attentionの重み2: {attention_weights_2}")

context_vec2 = attention_weights_2 @ values
print(f"コンテキストベクトル：{context_vec2}")
