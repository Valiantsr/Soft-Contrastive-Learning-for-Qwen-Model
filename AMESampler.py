import torch
from torch.utils.data import Sampler
import random

class LengthInBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=42, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Deteksi mode pairwise/triplet dari field di dataset
        sample = dataset[0]
        if "input_ids_1" in sample and "input_ids_2" in sample:
            self.mode = "pairwise"
            self.lengths = [
                max(len(example["input_ids_1"]), len(example["input_ids_2"]))
                for example in dataset
            ]
            if batch_size % 2 != 0:
                raise ValueError("Batch size harus genap untuk SCL (pairwise).")
        elif "anchor_input_ids" in sample and "positive_input_ids" in sample and "negative_input_ids" in sample:
            self.mode = "triplet"
            # Hitung panjang terpanjang dari anchor, positive, negative
            self.lengths = [
                max(
                    len(example["anchor_input_ids"]),
                    len(example["positive_input_ids"]),
                    len(example["negative_input_ids"])
                ) for example in dataset
            ]
            # Tidak perlu validasi batch_size
        else:
            raise ValueError("Dataset field tidak dikenali untuk LengthInBatchSampler.")
        self.total_indices = list(range(len(dataset)))

    def __iter__(self):
        rng = random.Random(self.seed)
        if self.shuffle:
            paired = list(zip(self.total_indices, self.lengths))
            rng.shuffle(paired)
            paired.sort(key=lambda x: x[1])
            sorted_indices = [idx for idx, _ in paired]
        else:
            sorted_indices = self.total_indices
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
        flat_indices = [idx for batch in batches for idx in batch]
        return iter(flat_indices)

    def __len__(self):
        return (len(self.dataset) // self.batch_size) * self.batch_size
