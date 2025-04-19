import torch
import torch.nn as nn

from chapter3.multi_head_attention_class import MultiHeadAttention


class DummyGPTModel(nn.Module):
    """ プレースホルダのGPTモデルクラス

    Notes:
        GPTModelクラスに変更
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # トークンID -> 埋め込みベクトル変換層
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # トークンの位置を表すベクトル
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # ドロップアウト層
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # TransformerBlockのプレースホルダ（後でSelf-Attentionなどを実装）
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最終正規化層のプレースホルダ（後でLayerNormを実装）
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # 各トークン位置の出力を語彙数次元のスコア（ロジット）に変換する線形層
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # トークンIDから埋め込みベクトル算出
        tok_embeds = self.tok_emb(in_idx)

        # 位置ベクトルを取得
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # 入力（トークンの埋め込みベクトル）に位置ベクトルを加算
        x = tok_embeds + pos_embeds

        # ドロップアウト
        x = self.drop_emb(x)

        # TransformerBlock
        x = self.trf_blocks(x)

        # 最終正規化
        x = self.final_norm(x)

        # 出力
        logits = self.out_head(x)
        return logits


class GPTModel(nn.Module):
    """ プレースホルダのGPTモデルクラス """

    def __init__(self, cfg: dict):
        super().__init__()

        # トークンID -> 埋め込みベクトル変換層
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # トークンの位置を表すベクトル
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # ドロップアウト層
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最終正規化層
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 各トークン位置の出力を語彙数次元のスコア（ロジット）に変換する線形層
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # トークンIDから埋め込みベクトル算出
        tok_embeds = self.tok_emb(in_idx)

        # 位置ベクトルを取得
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # 入力（トークンの埋め込みベクトル）に位置ベクトルを加算
        x = tok_embeds + pos_embeds

        # ドロップアウト
        x = self.drop_emb(x)

        # TransformerBlock
        x = self.trf_blocks(x)

        # 最終正規化
        x = self.final_norm(x)

        # 出力
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    """ TransformerBlockのプレースホルダクラス

    Notes:
        TransformerBlockクラスに変更
    """

    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            is_bias=cfg["qkv_bias"]
        )

        # フィードフォワード
        self.ff = FeedForward(cfg)

        # 正規化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # ショートカット接続用のドロップアウト層
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        # Attentionブロックのショートカット接続
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # フィードフォワードブロックのショートカット接続
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class DummyLayerNorm(nn.Module):
    """ LayerNormのプレースホルダクラス

    LayerNormクラスに変更
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        # ゼロ除算を防ぐ用
        self.eps = 1e-5
        # スケーリング
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # シフト
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 平均
        mean = x.mean(dim=-1, keepdim=True)
        # 分散
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 平均0, 分散1に正規化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """ GELU活性化関数

    Notes:
        PyTorch公式がGELU関数を提供しているが、自作したのは勉強用？
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi))) * (x + 0.44715 * torch.pow(x, 3)))


class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # 語彙のサイズ（数）
        "context_length": 1024,  # コンテキスト長さ
        "emb_dim": 768,  # 埋め込み次元数
        "n_heads": 12,  # Attentionヘッド数
        "n_layers": 12,  # 層の数
        "drop_rate": 0.1,  # ドロップアウト率
        "qkv_bias": False  # クエリ、キー、値のバイアスを使用するか
    }
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print(x.shape)
    print(output.shape)