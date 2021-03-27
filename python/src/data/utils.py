"""
General data utilities.
"""

from typing import Tuple, List
import torch


class SamplingDataset(torch.utils.data.IterableDataset):
    """
    A dataset sampling from multiple datasets, each with a different weight,
    and stopping when the first runs out.
    """

    def __init__(
        self, datasets_and_weights=List[Tuple[torch.utils.data.IterableDataset, float]]
    ):
        super(SamplingDataset, self).__init__()
        self._datasets, self._weights = zip(*datasets_and_weights)

    def __iter__(self):
        ds_iters = [iter(ds) for ds in self._datasets]

        try:
            while True:
                budget = torch.rand(1).item() * sum(self._weights)
                for weight, ds_iter in zip(self._weights, ds_iters):
                    budget -= weight
                    if budget <= 0:
                        yield next(ds_iter)
                        break
        except StopIteration:
            pass