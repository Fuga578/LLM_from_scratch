import torch


class NeuralNetwork(torch.nn.Module):
    """ ２つの隠れ層を持つ多層パーセプトロン（全結合ニューラルネットワーク）

    Args:
        num_inputs (int): 入力数
        num_outputs (int）: 出力数

    Attributes:
        # num_inputs (int): 入力数
        # num_outputs (int）: 出力数
        layers (torch.nn): ニューラルネットワーク層
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        # self.num_inputs = num_inputs
        # self.num_outputs = num_outputs

        self.layers = torch.nn.Sequential(
            # 隠れ層1
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 隠れ層2
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 出力層
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


# モデル生成
model = NeuralNetwork(50, 3)
print(model)

# 訓練可能なパラメータ数をチェック
# for p in model.parameters():
#     print(p.numel(), p)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total number of trainable model parameters: {num_params}")

# 計算
torch.manual_seed(123)
X = torch.rand((1, 50)) # 出力
out = model(X)
print(f"output: {out}")

# 訓練、逆伝播をおこなわずに計算（推論）
with torch.no_grad():
    out = model(X)
print(f"output: {out}")
