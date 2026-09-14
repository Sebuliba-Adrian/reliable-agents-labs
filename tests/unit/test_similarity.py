"""Tier 1: unit tests for cosine_similarity itself, no network, no model."""

import pytest

from reliable_agents_labs.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_zero_vector_raises():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])
