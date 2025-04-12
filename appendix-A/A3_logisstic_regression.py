import torch
import torch.nn.functional as F


# 正解ラベル
y = torch.tensor([1.0])

# 入力特徴量
x1 = torch.tensor([1.1])

# 重み
w1 = torch.tensor([2.2])

# バイアス
b = torch.tensor([0.0])

# 入力値
z = x1 * w1 + b

# 出力
a = torch.sigmoid(z)

# 誤差（バイナリクロスエントロピー）
loss = F.binary_cross_entropy(a, y)

print(a)
print(loss)
