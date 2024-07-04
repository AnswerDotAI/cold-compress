from abc import ABC, abstractmethod
import itertools
import os
import regex as re
import string
import sentencepiece as spm
import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    TypedDict,
)


default_device = "cuda" if torch.cuda.is_available() else "cpu"


def is_punc_id(text):
    # Define a regex pattern that matches any character that is not whitespace or punctuation
    pattern = rf"^[\s{re.escape(string.punctuation)}]*$"
    return bool(re.match(pattern, text))


class TokenizerInterface(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.vocab = None

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass

    @abstractmethod
    def bos_id(self):
        pass

    @abstractmethod
    def eos_id(self):
        pass

    @abstractmethod
    def get_terminator_ids(self):
        pass

    @abstractmethod
    def special_ids(self) -> List[List[int]]:
        pass

    @abstractmethod
    def __len__(self):
        pass

    def punctuation_ids(self):
        return [i for i, wp in enumerate(self.vocab) if is_punc_id(wp)]

    def get_vocab(self):
        assert (
            self.vocab is not None
        ), "Subclasses should set the vocab attribute during initialization."
        return self.vocab


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(str(model_path))
        self.terminator_ids = [self.processor.eos_id()]
        self.vocab = [
            self.processor.id_to_piece(id)
            for id in range(self.processor.get_piece_size())
        ]

    def addl_special_ids(self):
        # If llama-2 in model path, return special tokens for llama-2
        if "llama-2" in str(self.model_path).lower():
            special_tokens = ["[INST]", "[/INST]"]
        else:
            raise ValueError(f"Unknown model path: {self.model_path}")

        def _encode_special(token):
            ids = self.processor.EncodeAsIds(token)
            if len(ids) > 1:
                print(f"Special token {token} was tokenized into {len(ids)} tokens")
            return ids

        return list(map(_encode_special, special_tokens))

    def special_ids(self) -> List[List[int]]:
        # Some of the chat templates aren't given a singular special token so we return a list of lists
        return [
            [self.processor.bos_id()],
            [self.processor.eos_id()],
            *self.addl_special_ids(),
        ]

    def encode(self, prompt):
        if isinstance(prompt, dict):
            text = prompt['instruction']
            if "input" in prompt:
                text += "\n" + prompt['input']
        else:
            text = prompt
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

    def get_terminator_ids(self):
        return self.terminator_ids

    def __len__(self):
        return self.processor.get_piece_size()

class Llama3Wrapper(TokenizerInterface):
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

    def encode(self, prompt):
        if isinstance(prompt, dict):
            text = prompt['instruction']
            if "input" in prompt:
                text += "\n" + prompt['input']
        else:
            text = prompt
        return self.model.encode(text)

    def special_ids(self) -> List[List[int]]:
        # Some of the chat templates aren't given a singular special token so we return a list of lists
        return [[x] for x in list(sorted(self.special_tokens.values()))]

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def get_terminator_ids(self):
        return self.terminator_ids

    def __len__(self):
        return self.model.n_vocab

class Llama3GistWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path, gist_position="instruction_end"):
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
        ] + ["<|gist|>"]
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
        self.gist_id = self.special_tokens["<|gist|>"]
        self.gist_position = gist_position
        self.prompt_with_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        self.prompt_without_input = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
        self.vocab = [self.model.decode([i]) for i in range(self.model.n_vocab)]   

    def encode_prompt(self, text):
        return self.encode(text)

    def encode(self, prompt):
        if isinstance(prompt, str):
            prompt = {"instruction": prompt}
            return self.encode(prompt)
        prompt_template = self.prompt_with_input if 'input' in prompt else self.prompt_without_input
        if "input" not in prompt:
            prompt = prompt_template.format(instruction=prompt['instruction'] + "<|gist|>")
        elif self.gist_position == "instruction":
            prompt = prompt_template.format(instruction=prompt["instruction"] + "<|gist|>", input=prompt["input"])
        elif self.gist_position == "input":
            prompt = prompt_template.format(instruction=prompt["instruction"], input=prompt["input"] + "<|gist|>")
        elif self.gist_position == "instruction_and_input":
            prompt = prompt_template.format(instruction=prompt["instruction"] + "<|gist|>", input=prompt["input"] + "<|gist|>")
        return self.model.encode(prompt, allowed_special={'<|gist|>'})

    def special_ids(self) -> List[List[int]]:
        # Some of the chat templates aren't given a singular special token so we return a list of lists
        return [[x] for x in list(sorted(self.special_tokens.values()))]

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def get_terminator_ids(self):
        return self.terminator_ids
    
    def gist_token_id(self):
        return self.gist_id
    
    def __len__(self):
        return len(self.model.n_vocab)


class TokenizersWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.terminator_ids = [self.tokenizer.eos_token_id]
        self.vocab = [
            self.tokenizer.decode(i) for i in range(self.tokenizer.vocab_size)
        ]

    def special_ids(self) -> List[List[int]]:
        if hasattr(self.tokenizer, "special_token_ids"):
            return [[x] for x in self.tokenizer.special_token_ids]

        # Its likely a tokenizer that has a special_tokens_map attribute
        special_tokens_ = list(self.tokenizer.special_tokens_map.values())
        special_tokens = []
        for t in special_tokens_:
            if type(t) == list:
                special_tokens.extend(t)
            else:
                special_tokens.append(t)
        special_tokens = list(set(special_tokens))
        return [[self.tokenizer.convert_tokens_to_ids(t)] for t in special_tokens]

    def encode(self, prompt):
        if isinstance(prompt, dict):
            text = prompt['instruction']
            if "input" in prompt:
                text += "\n" + prompt['input']
        else:
            text = prompt
        return self.model.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def bos_id(self):
        return self.tokenizer.bos_token_id

    def eos_id(self):
        return self.tokenizer.eos_token_id

    def get_terminator_ids(self):
        return self.terminator_ids

    def __len__(self):
        return len(self.tokenizer)


def get_tokenizer(tokenizer_model_path, model_name, is_chat=False):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """
    model_name = str(model_name).lower()
    if "gist" in model_name:
        if "instruction-only" in model_name:
            return Llama3GistWrapper(tokenizer_model_path, gist_position="instruction")
        elif "input-only" in model_name:
            return Llama3GistWrapper(tokenizer_model_path, gist_position="input")
        elif "instruction-and-input" in model_name:
            return Llama3GistWrapper(tokenizer_model_path, gist_position="instruction_and_input")
        else:
            raise ValueError(f"Invalid gist model name: {model_name}")
    if "llama-3" in model_name:
        return (
            Llama3ChatFormat(tokenizer_model_path)
            if is_chat
            else Llama3Wrapper(tokenizer_model_path)
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


class Llama3ChatFormat(Llama3Wrapper):
    def __init__(self, model_path):
        super().__init__(model_path)

    def encode_header(self, message: Message) -> List[int]:
        return [
            self.special_tokens["<|start_header_id|>"],
            *self.encode(message["role"]),
            self.special_tokens["<|end_header_id|>"],
            *self.encode("\n\n"),
        ]

    def encode_prompt(self, prompt):
        if isinstance(prompt, dict):
            text = prompt['instruction']
            if "input" in prompt:
                text += "\n" + prompt['input']
        return self.encode_dialog_prompt([{"role": "user", "content": text}])

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

    def encode_prompt(self, prompt):
        if isinstance(prompt, dict):
            text = prompt['instruction']
            if "input" in prompt:
                text += prompt['input']
        ids = [self.bos_id()]
        ids += self.encode(Llama2ChatFormat.B_INST + "\n\n")
        ids += self.encode(prompt + " " + Llama2ChatFormat.E_INST)
        return ids


class TokenizersChatFormat(TokenizersWrapper):
    def __init__(self, model_path):
        super().__init__(model_path)

    def encode_prompt(self, prompt: str):
        if isinstance(prompt, dict):
            text = prompt['instruction']
            if "input" in prompt:
                text += "\n" + prompt['input']
        messages = [{"role": "user", "content": prompt}]
        return self.encode_dialog_prompt(messages)

    def encode_dialog_prompt(self, dialog: List[Message]) -> List[int]:
        text = self.tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )
        return self.encode(text)


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def encode(tokenizer, prompt, device=default_device, is_chat=True):
    if is_chat:
        tokens = tokenizer.encode_prompt(prompt)
        encoded = torch.tensor(tokens, dtype=torch.int, device=device)
    else:
        encoded = encode_tokens(tokenizer, prompt, device=device, bos=True)

    return encoded
