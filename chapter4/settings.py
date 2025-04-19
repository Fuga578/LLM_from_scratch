# 1億2,400万パラメータ設定
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 語彙のサイズ（数）
    "context_length": 1024, # コンテキスト長さ
    "emb_dim": 768,         # 埋め込み次元数
    "n_heads": 12,          # Attentionヘッド数
    "n_layers": 12,         # 層の数
    "drop_rate": 0.1,       # ドロップアウト率
    "qkv_bias": False       # クエリ、キー、値のバイアスを使用するか
}