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

# 2番目の埋め込みベクトル
query = inputs[1]

# 2番目の埋め込みベクトルに対するattentionスコア
attention_score_2 = torch.empty(inputs.shape[0])
for i, embedding_vec in enumerate(inputs):
    attention_score_2[i] = torch.dot(embedding_vec, query)
print(f"attentionスコア2：{attention_score_2}")

# ソフトマックス関数で正規化
attention_weights_2 = torch.softmax(attention_score_2, dim=0)
print(f"正規化したattentionスコア2：{attention_weights_2}, sum: {attention_weights_2.sum()}")

# コンテキストベクトルを算出
context_vec_2 = torch.zeros(query.shape)
for i, embedding_vec in enumerate(inputs):
    context_vec_2 += attention_weights_2[i] * embedding_vec
print(f"コンテキストベクトル2：{context_vec_2}")

# 全ての入力トークンのattention重みを算出
# attention_scores = torch.empty(6, 6)
# for i, xi in enumerate(inputs):
#     for j, xj in enumerate(inputs):
#         attention_scores[i, j] = torch.dot(xi, xj)
attention_scores = inputs @ inputs.T
attention_weights = torch.softmax(attention_scores, dim=-1)
context_vec = attention_weights @ inputs
print(f"全attentionスコア：{attention_scores}")
print(f"全attentionの重み：{attention_weights}, sum: {attention_weights.sum()}")
print(f"全コンテキストベクトル：{context_vec}")
