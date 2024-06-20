import itertools
import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    TypedDict,
)


class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path
        self.vocab = None

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_terminator_ids(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_vocab(self):
        assert self.vocab is not None, "Subclasses should set the vocab attribute during initialization."
        return self.vocab


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))
        self.terminator_ids = [self.processor.eos_id()]
        self.vocab = [self.processor.id_to_piece(id) for id in range(self.processor.get_piece_size())]

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

    def get_terminator_ids(self):
        return self.terminator_ids


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.terminator_ids = [self._eos_id, self.special_tokens["<|eot_id|>"]]
        self.vocab = [self.model.decode([i]) for i in range(self.model.n_vocab)]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def get_terminator_ids(self):
        return self.terminator_ids


class TokenizersWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.terminator_ids = [self.tokenizer.eos_token_id]
        self.vocab = [self.tokenizer.decode(i) for i in range(self.tokenizer.vocab_size)]

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def bos_id(self):
        return self.tokenizer.bos_token_id

    def eos_id(self):
        return self.tokenizer.eos_token_id

    def get_terminator_ids(self):
        return self.terminator_ids


def get_tokenizer(tokenizer_model_path, model_name, is_chat=False):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """
    if "llama-3" in str(model_name).lower():
        return (
            Llama3ChatFormat(tokenizer_model_path)
            if is_chat
            else TiktokenWrapper(tokenizer_model_path)
        )
    elif "llama-2" in str(model_name).lower():
        return (
            Llama2ChatFormat(tokenizer_model_path)
            if is_chat
            else SentencePieceWrapper(tokenizer_model_path)
        )
    else:
        return (
            TokenizersChatFormat(tokenizer_model_path)
            if is_chat
            else TokenizersWrapper(tokenizer_model_path)
        )


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class Llama3ChatFormat(TiktokenWrapper):
    def __init__(self, model_path):
        super().__init__(model_path)

    def encode_header(self, message: Message) -> List[int]:
        return [
            self.special_tokens["<|start_header_id|>"],
            *self.encode(message["role"]),
            self.special_tokens["<|end_header_id|>"],
            *self.encode("\n\n"),
        ]

    def encode_prompt(self, prompt: str):
        return self.encode_dialog_prompt([{"role": "user", "content": prompt}])

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(self.encode(message["content"].strip()))
        tokens.append(self.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: List[Message]) -> List[int]:
        return [
            self.special_tokens["<|begin_of_text|>"],
            *list(itertools.chain(*map(self.encode_message, dialog))),
            # Add the start of an assistant message for the model to complete.
            *self.encode_header({"role": "assistant", "content": ""}),
        ]


class Llama2ChatFormat(SentencePieceWrapper):
    B_INST = "[INST]"
    E_INST = "[/INST]"

    def __init__(self, model_path):
        super().__init__(model_path)

    def encode_prompt(self, prompt: str):
        ids = [self.bos_id()]
        ids += self.encode(Llama2ChatFormat.B_INST + "\n\n")
        ids += self.encode(prompt + " " + Llama2ChatFormat.E_INST)
        return ids


class TokenizersChatFormat(TokenizersWrapper):
    def __init__(self, model_path):
        super().__init__(model_path)

    def encode_prompt(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.encode_dialog_prompt(messages)

    def encode_dialog_prompt(self, dialog: List[Message]) -> List[int]:
        text = self.tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )
        return self.encode(text)
