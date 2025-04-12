import re


class SimpleTokenizerV1:
    """ シンプルなトークナイザクラス

    Args:
        vocab (dict[str, int]): {トークン: トークンID}の辞書

    Attributes:
        str_to_int (dict[str, int]): {トークン: トークンID}の辞書
        int_to_str (dict[int, str]):{トークンID: トークン}の辞書
    """

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab  # {トークン: トークンID}
        self.int_to_str = {i: s for s, i in vocab.items()}  # {トークンID: トークン}

    def encode(self, text: str) -> list[int]:
        """ 入力テキストをトークンIDに変換 """

        # テキストを分割（トークン化）
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # トークンに一致するトークンID取得
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids: list[int]) -> str:
        """ トークンIDからテキストを生成 """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """ コンテキストトークンに対応したトークナイザクラス

    <|endoftext|>, <|unk|>トークン対応

    Args:
        vocab (dict[str, int]): {トークン: トークンID}の辞書

    Attributes:
        str_to_int (dict[str, int]): {トークン: トークンID}の辞書
        int_to_str (dict[int, str]):{トークンID: トークン}の辞書
    """

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab  # {トークン: トークンID}
        self.int_to_str = {i: s for s, i in vocab.items()}  # {トークンID: トークン}

    def encode(self, text: str) -> list[int]:
        """ 入力テキストをトークンIDに変換 """

        # テキストを分割（トークン化）
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # 未知のトークンを<|unk|>トークンに置き換え
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        # トークンに一致するトークンID取得
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids: list[int]) -> str:
        """ トークンIDからテキストを生成 """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
