import random
from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset import (
    BucketingSampler,
    DynamicBucketingSampler,
    DynamicCutSampler,
    PerturbSpeed,
    PerturbVolume,
    RoundRobinSampler,
    SimpleCutSampler,
    ZipSampler,
)
from lhotse.testing.dummies import DummyManifest

SAMPLERS = [
    "simple",
    "round_robin",
    "zip",
    "bucketing",
    "dynamic",
    "dynamic_bucketing",
    "dynamic_indexed",
    "dynamic_bucketing_indexed",
]


@pytest.mark.parametrize("kind", SAMPLERS)
@pytest.mark.parametrize("prior_epochs", [0, 2])
@pytest.mark.parametrize("transform_seed", [None, 123])
def test_sampler_restores_batch_transform_state(
    tmp_path, kind, prior_epochs, transform_seed
):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=48).drop_features()
    path = tmp_path / "cuts.jsonl"
    cuts.to_jsonl(path)

    def make_sampler():
        kwargs = dict(max_cuts=4, shuffle=True, seed=42)
        if kind == "simple":
            sampler = SimpleCutSampler(cuts, **kwargs)
        elif kind in ("round_robin", "zip"):
            cls = RoundRobinSampler if kind == "round_robin" else ZipSampler
            sampler = cls(
                SimpleCutSampler(cuts.subset(first=24), **kwargs),
                SimpleCutSampler(cuts.subset(last=24), **kwargs),
            )
        elif kind == "bucketing":
            sampler = BucketingSampler(cuts, num_buckets=2, **kwargs)
        else:
            source = CutSet.from_file(path, indexed=kind.endswith("_indexed"))
            if kind.startswith("dynamic_bucketing"):
                sampler = DynamicBucketingSampler(
                    source, num_buckets=2, buffer_size=16, **kwargs
                )
            else:
                sampler = DynamicCutSampler(source, **kwargs)
        # Stateless transforms must not shift the stateful transforms' positions.
        sampler.map(lambda batch: batch)
        sampler.map(
            PerturbVolume(
                p=1.0,
                randgen=random.Random(transform_seed)
                if transform_seed is not None
                else None,
            )
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.5, randgen=random.Random(7)))
        return sampler

    sampler = make_sampler()
    for epoch in range(prior_epochs):
        sampler.set_epoch(epoch)
        list(sampler)
    sampler.set_epoch(prior_epochs)
    expected = []
    for index, batch in enumerate(sampler):
        if index == 2:
            state = deepcopy(sampler.state_dict())
        elif index > 2:
            expected.append(batch)

    restored = make_sampler()
    restored.load_state_dict(state)
    restored.set_epoch(prior_epochs)
    iter(restored)
    assert list(restored) == expected

    assert [t.state_dict() for t in restored._transforms[1:]] == [
        t.state_dict() for t in sampler._transforms[1:]
    ]
