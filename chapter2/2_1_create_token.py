import re


# テキスト読み込み
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"トータル文字数：{len(raw_text)}")
print(f"先頭100文字：{raw_text[:99]}")

# 単語で分割（トークン生成）
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]  # スペース除去
print(f"トークン：{preprocessed}")

# 重複トークンを削除し、アルファベット順にソート
all_words = sorted(set(preprocessed))
print(f"総ワード数：{len(all_words)}")

# {トークン：トークンID}の辞書作成
# この辞書から、新しいテキスト（トークン）に対してトークンIDを生成
vocab = {token:token_id for token_id, token in enumerate(all_words)}

# 先頭50件確認
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

