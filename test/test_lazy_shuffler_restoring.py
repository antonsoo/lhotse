import random
from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler
from lhotse.lazy import LazyShuffler
from lhotse.testing.dummies import DummyManifest


def _identity(cut):
    return cut


def _even_id(cut):
    return int(cut.id.rsplit("-", 1)[-1]) % 2 == 0


@pytest.mark.parametrize(
    "graph", ["indexed", "map_filter", "chain", "repeat", "nested"]
)
@pytest.mark.parametrize("buffer_size", [4, 100])
@pytest.mark.parametrize("remaining", [20, 3, 0])
def test_shuffle_checkpoint_preserves_later_iterations(
    tmp_path, graph, buffer_size, remaining
):
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=48).to_jsonl(path)

    def make_shuffler():
        cuts = CutSet.from_file(path, indexed=True)
        if graph == "map_filter":
            cuts = cuts.map(_identity).filter(_even_id)
        elif graph == "chain":
            cuts = cuts + CutSet.from_file(path, indexed=True)
        elif graph == "repeat":
            cuts = cuts.repeat(times=2)
        elif graph == "nested":
            cuts = cuts.shuffle(rng=random.Random(19), buffer_size=8)
        return LazyShuffler(cuts, buffer_size=buffer_size, rng=random.Random(42))

    total = len(list(make_shuffler()))
    original = make_shuffler()
    iterator = iter(original)
    for _ in range(total - remaining):
        next(iterator)
    state = deepcopy(original.state_dict())
    expected = list(iterator)

    restored = make_shuffler()
    restored.load_state_dict(state)
    assert list(restored) == expected
    # The source's pending EOF restoration must be consumed in this iteration.
    # Otherwise the first fresh iteration is empty, even though resume matched.
    for _ in range(2):
        expected = list(original)
        assert len(expected) == total
        assert list(restored) == expected


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("buffer_size", [4, 100])
def test_dynamic_sampler_next_epoch_after_shuffle_restore(
    tmp_path, indexed, shuffle, buffer_size
):
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=48).to_jsonl(path)

    def make_sampler():
        return DynamicCutSampler(
            CutSet.from_file(path, indexed=indexed),
            max_cuts=4,
            shuffle=shuffle,
            shuffle_buffer_size=buffer_size,
            seed=42,
        )

    original = make_sampler()
    expected = []
    for index, batch in enumerate(original):
        if index == 2:
            state = deepcopy(original.state_dict())
        elif index > 2:
            expected.append(batch)

    restored = make_sampler()
    restored.load_state_dict(state)
    assert list(restored) == expected
    for epoch in range(1, 3):
        original.set_epoch(epoch)
        restored.set_epoch(epoch)
        expected = list(original)
        assert len(expected) == 12
        assert list(restored) == expected
