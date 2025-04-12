import torch
from torch.utils.data import Dataset, DataLoader


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
print(len(train_ds))
# for x, y in train_ds:
#     print(x, y)

# データセットをデータローダーに読み込ませる
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # テストデータセットはシャッフルの必要なし
    num_workers=0
)

# データの確認（訓練エポックに該当）
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

# 訓練エポック最後のバッチを削除
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
