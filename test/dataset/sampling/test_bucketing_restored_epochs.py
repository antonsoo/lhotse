"""An exhausted source's pending restore must not consume a later epoch."""

from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.testing.dummies import DummyManifest


def identity(cut):
    return cut


def select_even(cut):
    return int(cut.id.rsplit("-", 1)[-1]) % 2 == 0


def remaining(sampler):
    batches = []
    while True:
        try:
            batches.append(tuple(next(sampler).ids))
        except StopIteration:
            return batches


@pytest.mark.parametrize("graph", ["plain", "map_filter", "repeat", "chain"])
@pytest.mark.parametrize("buffer_size", [11, 1000])
@pytest.mark.parametrize("progress", [3, 100])
@pytest.mark.parametrize("drop_last", [False, True])
def test_indexed_bucketing_restores_following_epochs(
    tmp_path, graph, buffer_size, progress, drop_last
):
    path = tmp_path / "cuts.jsonl"
    cuts = DummyManifest(CutSet, begin_id=0, end_id=41)
    CutSet.from_cuts(
        cut.truncate(duration=0.25 + 0.1 * (idx % 7), preserve_id=True)
        for idx, cut in enumerate(cuts)
    ).to_file(path)

    def make_sampler():
        cuts = CutSet.from_file(path, indexed=True)
        if graph == "map_filter":
            cuts = cuts.map(identity).filter(select_even)
        elif graph == "repeat":
            cuts = cuts.repeat(times=2)
        elif graph == "chain":
            cuts = cuts + CutSet.from_file(path, indexed=True)
        return DynamicBucketingSampler(
            cuts,
            max_cuts=3,
            duration_bins=[0.45, 0.65],
            buffer_size=buffer_size,
            shuffle=True,
            seed=17,
            drop_last=drop_last,
        )

    original = make_sampler()
    original.set_epoch(3)
    iter(original)
    for _ in range(progress):
        try:
            next(original)
        except StopIteration:
            break
    state = deepcopy(original.state_dict())
    expected = remaining(original)
    restored = make_sampler()
    restored.load_state_dict(state)
    iter(restored)
    assert remaining(restored) == expected
    for epoch in [4, 5]:
        original.set_epoch(epoch)
        restored.set_epoch(epoch)
        iter(original)
        iter(restored)
        assert remaining(restored) == remaining(original)


def test_indexed_bucketing_restores_following_epochs_after_full_buffer_stop(tmp_path):
    # With drop_last=True an epoch can end while the buckets are full and the source
    # still has data. Six short cuts fill the buffer and the first batch takes four
    # of them; the refill (three long, one short) leaves three cuts in each bucket,
    # so neither can yield a full batch and the remaining six cuts are never read.
    durations = [0.3] * 6 + [0.9, 0.9, 0.9, 0.3] + [0.3] * 6
    path = tmp_path / "cuts.jsonl"
    CutSet.from_cuts(
        cut.truncate(duration=duration, preserve_id=True)
        for cut, duration in zip(
            DummyManifest(CutSet, begin_id=0, end_id=len(durations)), durations
        )
    ).to_file(path)

    def make_sampler():
        return DynamicBucketingSampler(
            CutSet.from_file(path, indexed=True),
            max_cuts=4,
            duration_bins=[0.6],
            buffer_size=6,
            drop_last=True,
        )

    original = make_sampler()
    iter(original)
    first_epoch = remaining(original)
    assert len(first_epoch) == 1
    state = deepcopy(original.state_dict())

    restored = make_sampler()
    restored.load_state_dict(state)
    iter(restored)
    assert remaining(restored) == []
    for epoch in [1, 2]:
        restored.set_epoch(epoch)
        iter(restored)
        assert remaining(restored) == first_epoch
