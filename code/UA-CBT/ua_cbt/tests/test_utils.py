import pytest
from pytest import approx
import spacy
from ua_cbt.iteration import split_tales_by_story, _find_split_sizes

b = breakpoint


def test_splitting():
    ns = [
        [0.4],
        [0.4, 0.5],
    ]
    expected = [
        [0.4, 0.6],
        [0.4, 0.5, 0.1],
    ]

    for i, n in enumerate(ns):
        assert _find_split_sizes(n) == approx(expected[i])


def test_split_fail():
    ns = [
        [0.4, 0.6],
        [1.4],
        list(),
    ]

    for i, n in enumerate(ns):
        with pytest.raises(ValueError):
            _find_split_sizes(n)


def test_split_tales_by_story():
    ns = [
        [0.4],
        [0.4, 0.5],
    ]
    expected = [
        [4, 6],
        [4, 5, 1],
    ]
    stories = [[i] for i,s in enumerate(range(10))]

    for i, ss in enumerate(ns):
        splits = split_tales_by_story(stories, split_sizes=ss)
        split_sizes = [len(x) for x in splits]
        assert split_sizes == expected[i]

        # Should have lost no stories
        assert sum([len(x) for x in splits])==len(stories)

def test_split_tales_by_story_edge_cases():
    ns = [
        # Enough for one split but not the other
        [0.41, 0.51],
        # As well as larger than a single split
        [0.05]*11,
    ]

    # 9!
    stories = [[i] for i,s in enumerate(range(9))]

    for i, ss in enumerate(ns):
        with pytest.raises(ValueError):
            splits = split_tales_by_story(stories, split_sizes=ss)
