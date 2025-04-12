import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


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


class ToyDataset(Dataset):
    """" カスタムデータセットクラス

    Args:
        X (torch.tensor):   入力データ
        y (torch.tensor):   正解データ

    Attributes:
        features (torch.tensor):    入力データ
        labels (torch.tensor):  正解データ
    """

    def __init__(self, X: torch.tensor, y: torch.tensor):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """ データレコードと対応するラベルを一つだけ取得"""
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


torch.manual_seed(123)

# 訓練データ
# 入力
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

# 正解ラベル
y_train = torch.tensor([0, 0, 0, 1, 1])

# テストデータ
# 入力
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

# 正解ラベル
y_test = torch.tensor([0, 1])

# データセット生成
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # テストデータセットはシャッフルの必要なし
    num_workers=0
)

# モデル（特徴量2、正解ラベル2）
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# オプティマイザ（確率的勾配降下法）
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

# エポック数
num_epochs = 3

for epoch in range(num_epochs):

    # 学習モード
    model.train()

    # 学習
    for batch_idx, (features, labels) in enumerate(train_loader):

        # 出力
        logits = model(features)

        # 誤差（中身で、logitsをソフトマックス関数にかけている）
        loss = F.cross_entropy(logits, labels)

        # 勾配を累積したくないため、前のイテレーションの勾配を0に設定
        # （勾配は加算されてしまう）
        optimizer.zero_grad()

        # 損失の勾配計算
        loss.backward()

        # オプティマイザが勾配を使ってモデルのパラメータ更新
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}"
              f"Batch {batch_idx}/{len(train_loader)}"
              f"train loss: {loss:.2f}")

        # モデル評価
        model.eval()
        # TODO 評価する場合は以下に記述


# # テスト
# model.eval()
#
# with torch.no_grad():
#     train_outputs = model(X_train)
#     test_outputs = model(X_test)
# torch.set_printoptions(sci_mode=False)  # 出力結果を見やすく
# train_probas = torch.softmax(train_outputs, dim=1)
# test_probas = torch.softmax(test_outputs, dim=1)
# print(train_probas)
# print(test_probas)
#
# # 分類
# # （クラスラベルを得るだけならソフトマックス関数は不要）
# train_precitions = torch.argmax(train_probas, dim=1)
# test_predictions = torch.argmax(test_probas, dim=1)
# print(train_precitions, test_predictions)
# print(train_precitions == y_train)
# print(test_predictions == y_test)

def compute_accuracy(model: any, dataloader: DataLoader) -> int:
    """ 予測正解率を計算します。

    Args:
        model (any):    モデル
        dataloader (DataLoader):    データセットローダー

    Returns:
        int:    予測正解率
    """
    # 評価モード
    model.eval()

    correct = 0.0   # 正解数
    total_examples = 0  # データ数

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        print(predictions, labels)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))
