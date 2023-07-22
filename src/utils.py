from numpy import array, append, array_split
from typing import List


def data_transform():
    pass


def ml_partitions_indices(
    n: int,
    split_fractions: List[float]
) -> List[List[int]]:
    split_fractions = array(split_fractions)
    assert split_fractions.sum().round(1) == 1.0
    split_freq = append(
        t_v_freq := (n*split_fractions[:2]).astype(int),
        n-t_v_freq.sum()
    )
    assert n == split_freq.sum()
    return split_freq
