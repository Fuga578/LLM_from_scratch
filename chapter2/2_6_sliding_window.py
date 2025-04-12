import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """ スライディングウィンドウ対応データセット

    Args:
        txt (str):  テキスト
        tokenizer (any):    トークナイザ
        max_length (int):   シーケンス分割サイズ
        stride (int):   シフトサイズ（どれだけずらすか）

    Examples:
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_length = 5
        stride = 3
        の場合
        ・1回目（i = 0）
        　input_chunk = [1, 2, 3, 4, 5]
        　target_chunk = [2, 3, 4, 5, 6]
        ・2回目（i = 3）
        　input_chunk = [4, 5, 6, 7, 8]
        　target_chunk = [5, 6, 7, 8, 9]
        ・3回目（i = 6）
        　input_chunk = [7, 8, 9, 10]
        　max_length = 5に足りないので計算されない
    """

    def __init__(self, txt: str, tokenizer: any, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        # text -> token_id
        token_ids = tokenizer.encode(txt)

        # max
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(txt: str, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0):
    # トークナイザ生成
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# テキスト読み込み
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# データローダー生成
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(f"first_data: {first_batch}")
