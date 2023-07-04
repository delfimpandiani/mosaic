from typing import List, Tuple
import random
import math
from torch.utils.data import Dataset
from collections import defaultdict

def stratified_split(dataset: Dataset, labels: List, fraction: float, random_state: float = None) -> Tuple[Dataset, List, Dataset, List]:
    """
    Split the dataset into two dataset by taking into account the labels distribution.
    Implementation comes from https://gist.github.com/Alvtron/9b9c2f870df6a54fda24dbd1affdc254

    Args:
        dataset (Dataset): Input dataset
        labels (List): Labels of the dataset
        fraction (float): Fraction of the first element in the split.
        random_state (float, optional): Random state for the sampler. Defaults to None.

    Returns:
        Tuple[Dataset, List, Dataset, List]: A tuple the split of the dataset and its labels. The first two elements
            have size ~fraction, the last two elements `1 - ~fraction`.
    """
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels