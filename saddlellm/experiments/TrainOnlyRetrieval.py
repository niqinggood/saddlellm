"""Dependency-free BM25 retrieval with explicit split and group isolation."""

# ruff: noqa: N999 -- Matches the public module naming convention.

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    group: str
    text: str
    split: str = "train"


def _tokens(text):
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold())


class TrainOnlyBM25:
    def __init__(self, documents, *, heldout_groups=(), k1=1.5, b=0.75):
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("Retrieval corpus must not be empty")
        heldout = set(heldout_groups)
        if any(d.split != "train" or d.group in heldout for d in self.documents):
            raise ValueError(
                "Retrieval corpus contains a held-out group or non-training document"
            )
        if len({d.document_id for d in self.documents}) != len(self.documents):
            raise ValueError("Duplicate retrieval document IDs")
        self.counts = [Counter(_tokens(d.text)) for d in self.documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = max(1, sum(self.lengths) / len(self.lengths))
        self.frequency = Counter(t for c in self.counts for t in c)
        self.k1, self.b = k1, b

    def search(self, query, *, k=2, exclude_groups=()):
        if k < 1:
            raise ValueError("k must be positive")
        excluded = set(exclude_groups)
        scores = []
        for document, counts, length in zip(self.documents, self.counts, self.lengths):
            if document.group in excluded:
                continue
            score = 0.0
            for token in set(_tokens(query)):
                frequency = counts[token]
                if frequency:
                    idf = math.log(
                        1
                        + (len(self.documents) - self.frequency[token] + 0.5)
                        / (self.frequency[token] + 0.5)
                    )
                    score += (
                        idf
                        * frequency
                        * (self.k1 + 1)
                        / (
                            frequency
                            + self.k1 * (1 - self.b + self.b * length / self.average)
                        )
                    )
            if score > 0:
                scores.append((score, document))
        scores.sort(key=lambda pair: (-pair[0], pair[1].document_id))
        return scores[:k]
