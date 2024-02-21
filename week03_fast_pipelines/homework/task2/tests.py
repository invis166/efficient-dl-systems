import pytest

from .dataset import collate_fn, UNK_TOKEN, PAD_TOKEN


@pytest.fixture
def token_2_id():
    alphabet = ('a', 'b', 'c', 'd')
    token_2_id = dict(zip(alphabet, range(2, len(alphabet) + 2)))
    token_2_id[PAD_TOKEN] = 0
    token_2_id[UNK_TOKEN] = 1
    print(token_2_id)

    return token_2_id


def test_collate_fn_max_len(token_2_id):
    batch = [['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['a', 'b']]

    max_length = 4
    collated = collate_fn(batch, token_2_id, max_length=max_length)
    assert collated.shape == (len(batch), max_length)
