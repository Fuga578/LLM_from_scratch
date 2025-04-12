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


model = NeuralNetwork(2, 2)

# 保存（辞書型）
torch.save(model.state_dict(), "model.pth")

# 読み込み
# model1 = NeuralNetwork(3, 3)
model1 = NeuralNetwork(2, 2)
model1.load_state_dict(torch.load("model.pth"))
