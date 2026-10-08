from copy import deepcopy

import numpy as np
import pytest

from lhotse import CutSet
from lhotse.dataset import WeightedSimpleCutSampler
from lhotse.testing.dummies import DummyManifest


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("filter_cuts", [False, True])
def test_weighted_sampler_restore(shuffle, world_size, rank, filter_cuts):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=100)
    sampler = WeightedSimpleCutSampler(
        cuts,
        cuts_weight=list(range(1, 101)),
        num_samples=32,
        max_cuts=4,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        seed=42,
    )
    restored = WeightedSimpleCutSampler(
        cuts,
        cuts_weight=[1.0] * 100,
        num_samples=10,
        max_cuts=4,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        seed=42,
    )
    if filter_cuts:
        predicate = lambda cut: int(cut.id.split("-")[-1]) % 3 != 0
        sampler.filter(predicate)
        restored.filter(predicate)
    sampler.set_epoch(3)
    expected = []
    for idx, batch in enumerate(sampler):
        if idx == 1:
            state = deepcopy(sampler.state_dict())
            remaining = (sampler.remaining_cuts, sampler.remaining_duration)
        elif idx > 1:
            expected.append(batch)

    restored.load_state_dict(state)
    assert (restored.remaining_cuts, restored.remaining_duration) == remaining
    restored.set_epoch(3)
    assert list(restored) == expected
    assert restored.weights == sampler.weights
    assert restored.num_samples == sampler.num_samples

    restored.set_epoch(4)
    sampler.set_epoch(4)
    assert list(restored) == list(sampler)


@pytest.mark.parametrize("shuffle", [False, True])
def test_weighted_sampler_seed_and_epoch(shuffle):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=100)
    sampler = WeightedSimpleCutSampler(
        cuts,
        cuts_weight=[0.0] * 10 + [1.0] * 90,
        num_samples=32,
        max_cuts=4,
        seed=42,
        shuffle=shuffle,
    )
    # Unrelated global NumPy draws must not change the seeded sampler.
    original_rng_state = np.random.get_state()
    try:
        np.random.seed(123)
        first = list(sampler)
        np.random.seed(456)
        second = list(sampler)
    finally:
        np.random.set_state(original_rng_state)
    assert first == second
    assert all(int(c.id.split("-")[-1]) >= 10 for batch in first for c in batch)
    sampler.set_epoch(1)
    assert list(sampler) != first


@pytest.mark.parametrize("legacy", [False, True])
def test_weighted_sampler_restore_before_iteration(legacy):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=20)
    sampler = WeightedSimpleCutSampler(
        cuts,
        cuts_weight=[1.0] * 20,
        num_samples=10,
        max_cuts=3,
    )
    restored = deepcopy(sampler)
    state = sampler.state_dict()
    if legacy:
        state.pop("num_consumed", None)
    restored.load_state_dict(state)
    assert list(restored) == list(sampler)
