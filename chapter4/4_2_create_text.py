import tiktoken
import torch

from chapter4.settings import *
from chapter4.gpt_model import DummyGPTModel, GPTModel, generate_text_simple


# トークナイザ（テキスト <-> トークンID変換するもの）を生成
tokenizer = tiktoken.get_encoding("gpt2")

# テキスト
start_context = "Hello, I am"

# テキスト -> トークンID
encoded = tokenizer.encode(start_context)
print(f"エンコードされたトークンID: {encoded}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # バッチサイズ次元を追加
print(f"エンコードされたトークンID次元：{encoded_tensor.shape}")

model = GPTModel(GPT_CONFIG_124M)

model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print(f"出力：{out}")
print(f"推論後のトークン数：{len(out[0])}")

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"生成テキスト：{decoded_text}")
