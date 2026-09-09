import torch

from tiny_lm.dataset import LanguageModelingDataset


def test_dataset_returns_shifted_windows() -> None:
    dataset = LanguageModelingDataset(list(range(8)), block_size=4)
    input_ids, targets = dataset[2]
    assert len(dataset) == 4
    assert torch.equal(input_ids, torch.tensor([2, 3, 4, 5]))
    assert torch.equal(targets, torch.tensor([3, 4, 5, 6]))