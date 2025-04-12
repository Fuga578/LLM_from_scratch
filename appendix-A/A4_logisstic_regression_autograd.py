import torch
import torch.nn.functional as F


# 正解ラベル
y = torch.tensor([1.0])

# 入力特徴量
x1 = torch.tensor([1.1])

# 重み（更新パラメータ）
w1 = torch.tensor([2.2], requires_grad=True)

# バイアス（更新パラメータ）
b = torch.tensor([0.0], requires_grad=True)

# 入力値
z = x1 * w1 + b

# 出力
a = torch.sigmoid(z)

# 誤差（バイナリクロスエントロピー）
loss = F.binary_cross_entropy(a, y)

print(a)
print(loss)

# 損失の勾配
loss.backward()
print(w1.grad)
print(b.grad)
print(y.grad)
