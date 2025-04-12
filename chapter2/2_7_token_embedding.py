import torch


# 総トークン数
vocab_size = 6

# 埋め込み次元数
output_dim = 3

torch.manual_seed(123)

# 埋め込み層
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

input_tensor = torch.tensor([2, 3, 5, 2])

# 埋め込みベクトル生成
print(embedding_layer(input_tensor))
