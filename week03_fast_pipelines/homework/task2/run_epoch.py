import itertools
import statistics
import json
from collections import Counter
from enum import Enum
from functools import partial
from pathlib import Path
import time

from tqdm.auto import tqdm
from torchtext.data.utils import get_tokenizer
from dataset import (
    BigBrainDataset,
    BrainDataset,
    UltraDuperBigBrainDataset,
    UltraDuperBigBrainBatchSampler,
    collate_fn,
    PAD_TOKEN,
    UNK_TOKEN,
)
from transformer import TransformerModel, generate_square_subsequent_mask

import torch
from torch.utils.data import DataLoader


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model(ntoken) -> torch.nn.Module:
    return TransformerModel(
        ntoken=ntoken,
        d_model=1024,
        nhead=8,
        d_hid=2048,
        nlayers=1,
        dropout=0.1,
    )


def get_vocab(dataset, vocab_size: int):
    vocab_path = Path().cwd() / 'vocab.json'
    if vocab_path.exists():
        return json.loads(vocab_path.read_text())

    vocab = Counter()
    for line in dataset:
        vocab.update(line)

    word_2_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, (word, _) in enumerate(vocab.most_common(vocab_size), start=2):
        word_2_id[word] = i

    vocab_path.write_text(json.dumps(word_2_id))

    return word_2_id


def get_batch_processing_time_description(batch_processing_times):
    return (
        f'Min time: {min(batch_processing_times):.3f} '
        f'| Max time: {max(batch_processing_times):.3f} '
        f'| Mean time: {statistics.mean(batch_processing_times):.3f}'
        f'| Median time: {statistics.median(batch_processing_times):.3f}'
    )


def run_warmup_batch(model, dataloader, device):
    warmup_batch, _ = next(iter(dataloader))
    warmup_batch = warmup_batch.to(device)
    mask = generate_square_subsequent_mask(warmup_batch.shape[0]).to(device)
    model(warmup_batch, mask)


def measure_batch_processing_time(model, dataloader, device, n_batches=800):
    pbar = tqdm(itertools.islice(dataloader, n_batches), total=n_batches)
    batch_processing_times = []
    batch_lens = []
    for batch, target in pbar:
        start_time = time.perf_counter()

        batch = batch.to(device).transpose(1, 0)
        batch_lens.append(batch.shape[0])
        target = target.to(device)

        mask = generate_square_subsequent_mask(batch.shape[0]).to(device)
        model(batch, mask)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        batch_processing_times.append(end_time - start_time)

        pbar.set_description(get_batch_processing_time_description(batch_processing_times))

    print(statistics.mean(batch_lens), min(batch_lens), max(batch_lens))
    return batch_processing_times


def run_epoch(data_mode: DataMode) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer('basic_english')
    match data_mode:
        case DataMode.BRAIN:
            dataset = BrainDataset(
                data_path=Path().cwd() / 'data' / 'wikitext-103-raw' / 'wiki.train.raw',
                tokenizer=tokenizer
            )
            token_2_id = get_vocab(dataset, vocab_size=2048)
            collate_fn_partial = partial(collate_fn, max_length=dataset.max_length, token_2_id=token_2_id)
            dataloader = DataLoader(dataset=dataset, batch_size=128, collate_fn=collate_fn_partial, pin_memory=True)
        case DataMode.BIG_BRAIN:
            dataset = BigBrainDataset(
                data_path=Path().cwd() / 'data' / 'wikitext-103-raw' / 'wiki.train.raw',
                tokenizer=tokenizer
            )
            token_2_id = get_vocab(dataset, vocab_size=2048)
            collate_fn_partial = partial(collate_fn, token_2_id=token_2_id, max_length=None)
            dataloader = DataLoader(dataset=dataset, batch_size=128, collate_fn=collate_fn_partial, pin_memory=True)
        case DataMode.ULTRA_DUPER_BIG_BRAIN:
            dataset = UltraDuperBigBrainDataset(
                data_path=Path().cwd() / 'data' / 'wikitext-103-raw' / 'wiki.train.raw',
                tokenizer=tokenizer
            )
            token_2_id = get_vocab(dataset, vocab_size=2048)
            collate_fn_partial = partial(collate_fn, token_2_id=token_2_id, max_length=None)
            sampler = UltraDuperBigBrainBatchSampler(batch_size=128, dataset=dataset, bin_size=5)
            dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn_partial, pin_memory=True, batch_sampler=sampler)
        case _:
            raise NotImplementedError(f'Data mode "{data_mode}" is not supported')

    model = get_gpt2_model(ntoken=len(token_2_id)).to(device)
    model.train()

    run_warmup_batch(model, dataloader, device)
    measure_batch_processing_time(model, dataloader, device)


if __name__ == '__main__':
    torch.cuda.manual_seed(0)
    # BRAIN: Min time: 0.338 | Max time: 0.456 | Mean time: 0.353| Median time: 0.352
    # BIG_BRAIN: Min time: 0.050 | Max time: 0.363 | Mean time: 0.181| Median time: 0.175
    # ULTRA_DUPER_BIG_BRAIN_1: 0.002 | Max time: 0.180 | Mean time: 0.043| Median time: 0.035
    # 5: Min time: 0.002 | Max time: 0.222 | Mean time: 0.042| Median time: 0.032
    # 10: Min time: 0.002 | Max time: 0.240 | Mean time: 0.044| Median time: 0.035 
    # 20: Min time: 0.002 | Max time: 0.340 | Mean time: 0.052| Median time: 0.039
    # 50: Min time: 0.002 | Max time: 0.204 | Mean time: 0.047| Median time: 0.040
    # 640: Min time: 0.004 | Max time: 0.430 | Mean time: 0.114| Median time: 0.107
    run_epoch(DataMode.ULTRA_DUPER_BIG_BRAIN)
