import torch

from metalenvpool import memory_stats, resolve_device, synchronize


def test_resolve_auto_returns_torch_device():
    device = resolve_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "mps"}
    synchronize(device)


def test_memory_stats_shape():
    stats = memory_stats("auto")

    assert stats.device in {"cpu", "mps"}
    if stats.device == "cpu":
        assert stats.current_allocated_bytes is None
