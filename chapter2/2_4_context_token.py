import re
from tokenizer import SimpleTokenizerV2


# テキスト読み込み
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 単語で分割（トークン生成）
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]  # スペース除去

# 重複トークンを削除し、アルファベット順にソート
all_words = sorted(set(preprocessed))

# コンテキストトークン追加
# <|endoftext|> テキストソースの区切り
# <|unk|> 未知のトークン
all_words.extend(["<|endoftext|>", "<|unk|>"])

# {トークン：トークンID}の辞書作成
# この辞書から、新しいテキスト（トークン）に対してトークンIDを生成
vocab = {token:token_id for token_id, token in enumerate(all_words)}

# print(len(vocab))
# for item in vocab.items():
#     print(item)

# トークナイザ生成
tokenizer = SimpleTokenizerV2(vocab)

# テスト用テキスト
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(f"テストテキスト: {text}")

# text -> token_id
ids = tokenizer.encode(text)
print(f"トークンID: {ids}")

# token_id -> text
decode_text = tokenizer.decode(ids)
print(f"デコードテキスト：{decode_text}")
