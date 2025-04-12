import re
from tokenizer import SimpleTokenizerV1


# テキスト読み込み
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 単語で分割（トークン生成）
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]  # スペース除去

# 重複トークンを削除し、アルファベット順にソート
all_words = sorted(set(preprocessed))

# {トークン：トークンID}の辞書作成
# この辞書から、新しいテキスト（トークン）に対してトークンIDを生成
vocab = {token:token_id for token_id, token in enumerate(all_words)}

# トークナイザ生成
tokenizer = SimpleTokenizerV1(vocab)

# text -> token_id
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# token_id -> text
decode_text = tokenizer.decode(ids)
print(decode_text)

# 登録されていないテキストはトークンIDが生成できない
text = "Hello, do you like tea. Is this-- a test?"
tokenizer.encode(text)
