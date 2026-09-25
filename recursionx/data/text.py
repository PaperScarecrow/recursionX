"""Real-text data pipeline: tokenizers, pre-tokenised token bins, LM batches.

* :class:`ByteTokenizer` – dependency-free byte-level tokenizer.  Ids
  ``0..n_special-1`` are reserved (PAD/BOS/SEP/EOS + instruction/task tokens,
  matching ``recursionx.data.tasks``), bytes follow.
* :func:`load_tokenizer` – ``"byte"`` or ``"hf:<name>"`` (Hugging Face
  ``tokenizers``/``transformers``, optional dependency, loaded lazily).
* :func:`build_bin` – tokenise ``.txt``/``.md``/``.jsonl`` files into a flat
  ``uint16``/``uint32`` token file (nanoGPT-style), documents separated by EOS.
* :class:`TokenBin` – memory-mapped random-window sampler for training.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .tasks import BOS, EOS, PAD, SEP

N_SPECIAL = 32


class ByteTokenizer:
    name = "byte"

    def __init__(self, n_special: int = N_SPECIAL):
        self.n_special = n_special
        self.pad_id, self.bos_id, self.sep_id, self.eos_id = PAD, BOS, SEP, EOS

    @property
    def vocab_size(self) -> int:
        return self.n_special + 256

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = [b + self.n_special for b in text.encode("utf-8")]
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int]) -> str:
        bs = bytes(i - self.n_special for i in ids if self.n_special <= i < self.n_special + 256)
        return bs.decode("utf-8", errors="replace")


class HFTokenizer:
    """Thin wrapper around a Hugging Face tokenizer (optional dependency)."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer  # lazy, optional
        self.tok = AutoTokenizer.from_pretrained(name)
        self.name = f"hf:{name}"
        self.eos_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
        self.bos_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.eos_id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode(list(ids))


def load_tokenizer(spec: str = "byte"):
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer {spec!r}")


def iter_documents(paths: Sequence[str], jsonl_field: str = "text") -> Iterator[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.endswith((".txt", ".md", ".jsonl"))]
        else:
            files.append(p)
    for f in files:
        if f.endswith(".jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line).get(jsonl_field, "")
        else:
            with open(f, encoding="utf-8", errors="replace") as fh:
                yield fh.read()


def build_bin(paths: Sequence[str], out_path: str, tokenizer=None, val_frac: float = 0.01) -> Tuple[str, str]:
    """Tokenise documents into ``out_path`` (train) and ``out_path + '.val'``."""
    tok = tokenizer or ByteTokenizer()
    dtype = np.uint16 if tok.vocab_size < 65535 else np.uint32
    ids: List[int] = []
    for doc in iter_documents(paths):
        ids += tok.encode(doc, bos=True, eos=True)
    arr = np.asarray(ids, dtype=dtype)
    n_val = int(len(arr) * val_frac)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    arr[: len(arr) - n_val].tofile(out_path)
    arr[len(arr) - n_val:].tofile(out_path + ".val")
    with open(out_path + ".meta.json", "w") as f:
        json.dump({"tokenizer": tok.name, "vocab_size": tok.vocab_size, "dtype": np.dtype(dtype).name,
                   "n_train": int(len(arr) - n_val), "n_val": int(n_val)}, f)
    return out_path, out_path + ".val"


class TokenBin:
    """Random fixed-length windows from a flat token file."""

    def __init__(self, path: str, seq_len: int, dtype: Optional[str] = None, seed: int = 0):
        meta_path = (path[:-4] if path.endswith(".val") else path) + ".meta.json"
        if dtype is None and os.path.exists(meta_path):
            dtype = json.load(open(meta_path))["dtype"]
        self.data = np.memmap(path, dtype=np.dtype(dtype or "uint16"), mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        assert len(self.data) > seq_len + 1, f"{path}: not enough tokens for seq_len={seq_len}"

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, device="cpu"):
        """Returns (inp, tgt, weight) like ``tasks.collate``: next-token LM loss
        on every position except padding."""
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([np.asarray(self.data[s: s + self.seq_len + 1], dtype=np.int64) for s in starts])
        t = torch.from_numpy(x)
        inp, tgt = t[:, :-1], t[:, 1:]
        w = (tgt != PAD).float()
        return inp.to(device), tgt.to(device), w.to(device)
