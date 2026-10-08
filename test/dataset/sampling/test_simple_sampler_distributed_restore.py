from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import SimpleCutSampler
from lhotse.testing.dummies import DummyManifest


@pytest.fixture(params=[False, True], ids=["eager", "lazy"])
def cuts(request, tmp_path):
    cuts = CutSet.from_cuts(
        cut.truncate(duration=0.2 * (index % 5 + 1), preserve_id=True)
        for index, cut in enumerate(DummyManifest(CutSet, begin_id=0, end_id=100))
    )
    if request.param:
        path = tmp_path / "cuts.jsonl.gz"
        cuts.to_file(path)
        cuts = CutSet.from_file(path)
        assert cuts.is_lazy
    return cuts


@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1), (3, 2)])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("filter_cuts", [False, True])
@pytest.mark.parametrize("constraint", [{"max_cuts": 4}, {"max_duration": 2.5}])
def test_simple_sampler_distributed_restore(
    cuts, world_size, rank, shuffle, filter_cuts, constraint
):
    def make_sampler():
        sampler = SimpleCutSampler(
            cuts, **constraint, shuffle=shuffle, world_size=world_size, rank=rank
        )
        if filter_cuts:
            sampler.filter(lambda cut: cut.duration > 0.2)
        return sampler

    sampler = make_sampler()
    sampler.set_epoch(3)
    expected = []
    for index, batch in enumerate(sampler):
        if index == 1:
            state = deepcopy(sampler.state_dict())
            remaining = (sampler.remaining_cuts, sampler.remaining_duration)
        elif index > 1:
            expected.append(batch)

    restored = make_sampler()
    restored.load_state_dict(state)
    assert restored.remaining_cuts == remaining[0]
    assert restored.remaining_duration == pytest.approx(remaining[1])
    restored.set_epoch(3)
    assert list(restored) == expected
    assert restored.diagnostics == sampler.diagnostics

    # A later epoch must reset the consumed position along with the source.
    sampler.set_epoch(4)
    restored.set_epoch(4)
    assert list(restored) == list(sampler)
    assert restored.state_dict() == sampler.state_dict()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_simple_sampler_distributed_restore_at_end(cuts, rank, drop_last, exhausted):
    def make_sampler():
        return SimpleCutSampler(
            cuts, max_cuts=7, world_size=2, rank=rank, drop_last=drop_last
        )

    sampler = make_sampler()
    for batch in sampler:
        state = deepcopy(sampler.state_dict())
    if exhausted:
        state = sampler.state_dict()
    restored = make_sampler()
    restored.load_state_dict(state)
    assert list(restored) == []


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_batches", [0, 2])
def test_simple_sampler_legacy_restore(cuts, shuffle, num_batches):
    sampler = SimpleCutSampler(cuts, max_cuts=4, shuffle=shuffle)
    sampler.set_epoch(3)
    iter(sampler)
    for _ in range(num_batches):
        next(sampler)
    state = deepcopy(sampler.state_dict())
    state.pop("num_consumed", None)
    expected = []
    while True:
        try:
            expected.append(next(sampler))
        except StopIteration:
            break
    restored = SimpleCutSampler(cuts, max_cuts=4, shuffle=shuffle)
    restored.load_state_dict(state)
    assert list(restored) == expected
