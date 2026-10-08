from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import RoundRobinSampler, SimpleCutSampler
from lhotse.testing.dummies import DummyManifest

CUTS = DummyManifest(CutSet, begin_id=0, end_id=40)
CUTS_ALT = CUTS.modify_ids(lambda cid: cid + "_alt")


def make_sampler(randomize):
    return RoundRobinSampler(
        SimpleCutSampler(CUTS, max_cuts=2, shuffle=True),
        SimpleCutSampler(CUTS_ALT, max_cuts=3, shuffle=True),
        SimpleCutSampler(CUTS.subset(first=10), max_cuts=2, shuffle=True),
        randomize=randomize,
        seed=42,
    )


def drain(sampler):
    # iter() on a sampler that was not just restored starts the epoch over.
    batches = []
    while True:
        try:
            batches.append(list(next(sampler).ids))
        except StopIteration:
            return batches


@pytest.mark.parametrize("randomize", [True, [0.5, 0.3, 0.2]])
@pytest.mark.parametrize("epoch", [0, 3])
@pytest.mark.parametrize("steps", [0, 4, 15])
def test_round_robin_restore_randomized(randomize, epoch, steps):
    sampler = make_sampler(randomize)
    sampler.set_epoch(epoch)
    it = iter(sampler)
    for _ in range(steps):
        next(it)
    state = deepcopy(sampler.state_dict())
    expected = drain(it)

    restored = make_sampler(randomize)
    restored.load_state_dict(state)
    assert drain(iter(restored)) == expected


def test_round_robin_restore_then_discard_progress():
    sampler = make_sampler(True)
    it = iter(sampler)
    for _ in range(4):
        next(it)
    state = deepcopy(sampler.state_dict())
    first_epoch = drain(iter(make_sampler(True)))

    restored = make_sampler(True)
    restored.load_state_dict(state)
    restored.allow_iter_to_reset_state()
    assert drain(iter(restored)) == first_epoch


def test_round_robin_restore_checkpoint_without_rng_state():
    sampler = make_sampler(True)
    state = sampler.state_dict()
    state.pop("rng_state", None)  # checkpoints saved before this field existed

    restored = make_sampler(True)
    restored.load_state_dict(state)
    assert drain(iter(restored)) == drain(iter(sampler))
