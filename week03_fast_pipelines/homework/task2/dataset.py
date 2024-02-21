from collections import defaultdict
import itertools
from pathlib import Path
import random
from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler


MAX_LENGTH = 640
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


class BrainDataset(Dataset):
    '''Reads dataset by lines, truncates all the sequences up to MAX_LENGTH'''
    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        self.data_path = data_path
        self.max_length = max_length
        self.lines = self._read_lines(data_path, tokenizer, max_length)

    def __getitem__(self, idx: int):
        return self.lines[idx]

    def __len__(self):
        return len(self.lines)

    @staticmethod
    def _read_lines(data_path, tokenizer, max_length):
        with open(data_path, encoding='utf-8') as f:
            lines = list(filter(len, (tokenizer(line.rstrip())[:max_length] for line in f.readlines())))

        return lines


class BigBrainDataset(BrainDataset):
    pass


class UltraDuperBigBrainDataset(BigBrainDataset):
    pass


def collate_fn(
    batch: list[list[str]], token_2_id: dict[str, int], max_length: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    result = []
    batch_max_len = max(map(len, batch))
    for line in batch:
        pad_idx = token_2_id[PAD_TOKEN]
        if max_length:
            padding = [pad_idx] * (max_length - len(line))
        else:
            padding = [pad_idx] * (batch_max_len - len(line))

        unk_idx = token_2_id[UNK_TOKEN]
        ids = [token_2_id.get(token, unk_idx) for token in line] + padding
        result.append(ids)

    batch = torch.tensor(result)
    target = batch[:, 1:]

    return batch, target


class UltraDuperBigBrainBatchSampler(Sampler):
    def __init__(self, batch_size: int, dataset: UltraDuperBigBrainDataset, bin_size: int):
        self.batch_size = batch_size
        self.bin_size = bin_size
        self._dataset_size = len(dataset)

        self._idx_to_sample_len = dict(enumerate(map(len, dataset.lines)))
        self._len_to_indices = self._get_len_to_indices(self._idx_to_sample_len)

        self._indices = list(range(self._dataset_size))
        random.shuffle(self._indices)

    def __len__(self):
        return self._dataset_size // self.batch_size

    def __iter__(self):
        for sample_idx in self._indices:
            sample_len = self._idx_to_sample_len[sample_idx]
            yield self._collect_batch_indices_from_sample_len(sample_idx, sample_len)

    def _collect_batch_indices_from_sample_len(self, sample_idx, sample_len):
        batch_indices = [sample_idx]
        possible_lengths = list(range(sample_len - self.bin_size // 2, sample_len + self.bin_size // 2 + 1))
        random.shuffle(possible_lengths)
        for length in possible_lengths:
            texts = self._len_to_indices.get(length, [])
            batch_indices += random.sample(texts, k=min(self.batch_size, len(texts)))
            if len(batch_indices) >= self.batch_size:
                return batch_indices[:self.batch_size]

        return batch_indices

    @staticmethod
    def _get_len_to_indices(idx_to_sample_len):
        len_to_indices = defaultdict(list)
        for sample_idx, sample_len in idx_to_sample_len.items():
            len_to_indices[sample_len].append(sample_idx)

        return dict(len_to_indices)
