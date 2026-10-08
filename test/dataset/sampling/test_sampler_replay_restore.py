"""
Restoring a dynamic sampler without an indexed manifest replays the batches it had
yielded. Cuts rejected by ``.filter()`` and a dropped last batch are counted as
"discarded batches" in the diagnostics, but they are not batches to replay.
"""

from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.testing.dummies import DummyManifest


def not_every_fifth(cut):
    return int(cut.id.rsplit("-", 1)[-1]) % 5 != 0


def remaining(sampler_iter):
    batches = []
    while True:
        try:
            batches.append(list(next(sampler_iter).ids))
        except StopIteration:
            return batches


@pytest.fixture
def cuts_path(tmp_path):
    path = tmp_path / "cuts.jsonl.gz"
    DummyManifest(CutSet, begin_id=0, end_id=40).to_file(path)
    return path


def make_sampler(sampler_type, path, **kwargs):
    if sampler_type is DynamicBucketingSampler:
        kwargs["num_buckets"] = 2
    return sampler_type(CutSet.from_file(path), **kwargs)


@pytest.mark.parametrize(
    "sampler_type", [DynamicCutSampler, DynamicBucketingSampler], ids=["cut", "bucket"]
)
@pytest.mark.parametrize("num_batches", [1, 3, 6])
def test_replay_restore_with_sampler_filter(cuts_path, sampler_type, num_batches):
    def make():
        return make_sampler(sampler_type, cuts_path, max_cuts=4).filter(not_every_fifth)

    sampler = make()
    sampler_iter = iter(sampler)
    for _ in range(num_batches):
        next(sampler_iter)
    state = deepcopy(sampler.state_dict())
    assert sampler.diagnostics.current_epoch_stats.discarded_cuts > 0
    expected = remaining(sampler_iter)
    assert len(expected) == 8 - num_batches

    restored = make()
    restored.load_state_dict(state)
    assert remaining(iter(restored)) == expected


@pytest.mark.parametrize(
    "sampler_type", [DynamicCutSampler, DynamicBucketingSampler], ids=["cut", "bucket"]
)
def test_replay_restore_after_dropped_last_batch(cuts_path, sampler_type):
    def make():
        return make_sampler(sampler_type, cuts_path, max_cuts=3, drop_last=True)

    sampler = make()
    assert len(remaining(iter(sampler))) == 13  # 40 cuts: one cut is dropped
    state = deepcopy(sampler.state_dict())

    restored = make()
    restored.load_state_dict(state)
    assert remaining(iter(restored)) == []
    restored.set_epoch(1)
    assert len(remaining(iter(restored))) == 13
