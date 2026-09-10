import functools
import typing

def _convert_word_to_key(word: str) -> str:
    return word.strip().lower()


class SimpleTokenizer:
    _SEED_ID = 4

    def __init__(self) -> None:
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        built_in_tokens = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }

        vocab_tokens = functools.reduce(
            lambda acc, pair: {**acc, pair[1]: pair[0] + self._SEED_ID},
            enumerate(sorted({_convert_word_to_key(word=word) for text in texts for word in text.split()})),
            typing.cast("dict[str, int]", {})
        )

        self.word_to_id = {
            **built_in_tokens,
            **vocab_tokens
        }

        self.id_to_word = {
            _id: word for word, _id in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        return [
            self.word_to_id.get(
                _convert_word_to_key(word=word),
                self.word_to_id[self.unk_token]
            )
            for word in text.split()
        ]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join([
            self.id_to_word.get(
                _id,
                self.unk_token
            )
            for _id in ids
        ])