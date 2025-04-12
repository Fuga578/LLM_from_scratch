import tiktoken


# トークナイザ生成
tokenizer = tiktoken.get_encoding("gpt2")

# テキスト読み込み
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

encode_text = tokenizer.encode(raw_text)
print(len(encode_text))

context_size = 4
x = encode_text[:context_size]
y = encode_text[1:context_size+1]
print(f"x: {x}")
print(f"y:     {y}")

for i in range(1, context_size+1):
    context = encode_text[:i]
    desired = encode_text[i]
    print(f"{context} ----> {desired}")

for i in range(1, context_size+1):
    context = encode_text[:i]
    desired = encode_text[i]
    print(f"{tokenizer.decode(context)} ----> {tokenizer.decode([desired])}")
