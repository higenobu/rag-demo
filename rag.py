from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math
import re


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    # 日本語も雑に扱えるように、単純な単語/記号分割に寄せる
    return re.findall(r"\w+|[一-龠ぁ-んァ-ンー]+", text)


def simple_chunk(
    text: str,
    doc_id: str,
    chunk_size: int = 600,
    overlap: int = 120,
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(doc_id=doc_id, chunk_id=chunk_id, text=piece))
            chunk_id += 1
        if end >= len(text):
            break
        start += step

    return chunks


class LocalVectorIndex:
    """
    最小実装:
    - ファイルを chunk 化して保持
    - ベクトルDBの代わりに単純な token overlap / cosine 風スコアで検索
    """

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.chunk_tokens: List[List[str]] = []

    def add_text(
        self,
        text: str,
        doc_id: str,
        chunk_size: int = 600,
        overlap: int = 120,
    ) -> None:
        new_chunks = simple_chunk(
            text=text,
            doc_id=doc_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in new_chunks:
            self.chunks.append(chunk)
            self.chunk_tokens.append(tokenize(chunk.text))

    def add_file(
        self,
        path: str | Path,
        chunk_size: int = 600,
        overlap: int = 120,
    ) -> None:
        p = Path(path)
        if not p.is_file():
            return

        # テキストとして読めるものだけ対象
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return
        except Exception:
            return

        self.add_text(
            text=text,
            doc_id=p.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def add_corpus_dir(
        self,
        corpus_dir: str | Path,
        chunk_size: int = 600,
        overlap: int = 120,
        glob_pattern: str = "*",
    ) -> None:
        root = Path(corpus_dir)
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Corpus path is not a directory: {root}")

        files = sorted([p for p in root.rglob(glob_pattern) if p.is_file()])
        for p in files:
            self.add_file(p, chunk_size=chunk_size, overlap=overlap)

    def _score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        q_counts = {}
        d_counts = {}

        for t in query_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1
        for t in doc_tokens:
            d_counts[t] = d_counts.get(t, 0) + 1

        common = set(q_counts) & set(d_counts)
        dot = sum(q_counts[t] * d_counts[t] for t in common)
        q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
        d_norm = math.sqrt(sum(v * v for v in d_counts.values()))

        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if top_k <= 0:
            return []

        query_tokens = tokenize(query)
        scored: List[Tuple[Chunk, float]] = []

        for chunk, tokens in zip(self.chunks, self.chunk_tokens):
            score = self._score(query_tokens, tokens)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
