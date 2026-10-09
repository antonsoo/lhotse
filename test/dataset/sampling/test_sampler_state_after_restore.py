"""
A state saved after ``load_state_dict()`` and before the next batch has to keep
the restored position: the restore itself is deferred until iteration starts.
"""

import pickle
import random
from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.dataset.cut_transforms import PerturbVolume
from lhotse.testing.dummies import DummyManifest


def volumes(batch):
    return [
        (cut.id, tuple(round(t.factor, 6) for t in cut.recording.transforms or []))
        for cut in batch
    ]


def remaining(sampler_iter):
    batches = []
    while True:
        try:
            batches.append(volumes(next(sampler_iter)))
        except StopIteration:
            return batches


@pytest.mark.parametrize(
    "sampler_type", [DynamicCutSampler, DynamicBucketingSampler], ids=["cut", "bucket"]
)
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("call_iter", [False, True])
@pytest.mark.parametrize("with_transform", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_indexed_sampler_state_saved_right_after_restore(
    tmp_path, sampler_type, shuffle, call_iter, with_transform, reuse
):
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=40).to_file(path)

    def make_sampler():
        kwargs = {"num_buckets": 2} if sampler_type is DynamicBucketingSampler else {}
        sampler = sampler_type(
            CutSet.from_file(path, indexed=True),
            max_cuts=4,
            shuffle=shuffle,
            seed=0,
            **kwargs,
        )
        if with_transform:
            sampler.map(
                PerturbVolume(p=0.5, randgen=random.Random(7), preserve_id=True)
            )
        return sampler

    sampler = make_sampler()
    sampler_iter = iter(sampler)
    for _ in range(3):
        next(sampler_iter)
    first_state = deepcopy(sampler.state_dict())
    expected = remaining(sampler_iter)
    assert len(expected) == 7

    restored = make_sampler()
    if reuse:
        restored_iter = iter(restored)
        next(restored_iter)
        next(restored_iter)
    restored.load_state_dict(first_state)
    if call_iter:
        iter(restored)
    second_state = deepcopy(restored.state_dict())

    restored_again = make_sampler()
    restored_again.load_state_dict(second_state)
    assert [volumes(batch) for batch in restored_again] == expected
    # Saving must not disturb the sampler it was taken from either.
    assert [volumes(batch) for batch in restored] == expected


@pytest.mark.parametrize("indexed", [False, True])
def test_used_bucketing_sampler_is_picklable_after_restore(tmp_path, indexed):
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=40).to_file(path)
    sampler = DynamicBucketingSampler(
        CutSet.from_file(path, indexed=indexed),
        max_cuts=4,
        num_buckets=2,
        shuffle=True,
        seed=0,
    )
    sampler_iter = iter(sampler)
    for _ in range(3):
        next(sampler_iter)
    checkpoint = deepcopy(sampler.state_dict())
    expected = remaining(sampler_iter)

    sampler.load_state_dict(checkpoint)
    # Restored samplers must be serializable for spawned DataLoader workers.
    restored = pickle.loads(pickle.dumps(sampler))
    assert [volumes(batch) for batch in restored] == expected
