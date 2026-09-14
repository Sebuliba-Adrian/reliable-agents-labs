"""Chapter 11: the actual mechanism behind "semantic search," by hand,
before chapter 12 hands this exact math over to Qdrant. Nothing here is
provided by a library, on purpose, the point is to see plainly what a
vector database does under the hood, so it stops looking like magic.
"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """The cosine of the angle between two vectors: 1.0 for identical
    direction, 0.0 for perpendicular (unrelated), negative for opposite.
    Embedding vectors that mean similar things point in similar
    directions, this is the whole trick "semantic search" rests on.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must be the same length, got {len(a)} and {len(b)}")
    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cannot compute similarity against a zero vector")
    return dot_product / (magnitude_a * magnitude_b)
