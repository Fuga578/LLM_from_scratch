import tiktoken
import torch

from chapter4.settings import *
from chapter4.gpt_model import DummyGPTModel, GPTModel


# トークナイザ（テキスト <-> トークンID変換するもの）を生成
tokenizer = tiktoken.get_encoding("gpt2")

# トークンID保持用リスト
batch = []

# サンプルテキスト
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# トークンID生成
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch, batch.shape)

torch.manual_seed(123)

# モデル生成
# model = DummyGPTModel(GPT_CONFIG_124M)
model = GPTModel(GPT_CONFIG_124M)

# 推論
logits = model(batch)

# [2, 4, 50257]
# 2: バッチサイズ（txt1, txt2）
# 4: txt1, txt2の単語数（トークン数）
# 50257: 出力トークン候補数
# logitsをソフトマックス関数などにかけることで、最も可能性の高い次のトークンIDを算出
print(logits.shape)
print(logits)

# パラメータ数
total_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数：{total_params:,}")
