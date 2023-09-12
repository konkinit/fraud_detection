from numpy import array, append, random
from pandas import DataFrame, concat
from torch import device, cuda
from typing import List


def data_transform():
    pass


def get_device(device_id: int = 0) -> device:
    """Get device either CPU or GPU if available

    Args:
        device_id (int, optional): device id in case
        of multiple GPU. Defaults to 0.

    Returns:
        device: pytoorch device
    """
    if cuda.is_available():
        if cuda.device_count() == 1:
            return device("cuda")
        return device(f"cuda:{device_id}")
    return device("cpu")


def ml_partitions_indices(
    n: int,
    split_fractions: List[float]
) -> List[List[int]]:
    """Get train/validation/test split indices

    Args:
        n (int): number of samples in the dataset
        split_fractions (List[float]): split fractions

    Returns:
        List[List[int]]: 3-sized list of lists, each element
        representing the indices of each partition
    """
    split_fractions = array(split_fractions)
    assert split_fractions.sum().round(1) == 1.0
    split_freq = append(
        t_v_freq := (n*split_fractions[:2]).astype(int),
        n-t_v_freq.sum()
    )
    assert n == split_freq.sum()
    idx_array = array(range(n))
    random.shuffle(idx_array)
    return [
        idx_array[:split_freq[0]],
        idx_array[split_freq[0]:split_freq[0]+split_freq[1]],
        idx_array[-split_freq[2]:]
    ]


def losses_dataframe(
    n_epoch: int,
    training_losses: list,
    validation_losses: list
) -> DataFrame:
    """Get a dataframe gathering losses for each split and
    epoch

    Args:
        n_epoch (int): number of epoch
        training_losses (list): losses on train split
        validation_losses (list): losses on validation split

    Returns:
        DataFrame: dataframe object containing loss values
    """
    return concat(
        [
            DataFrame(
                {
                    "epoch": range(n_epoch),
                    "split": "train",
                    "loss": training_losses
                }
            ),
            DataFrame(
                {
                    "epoch": range(n_epoch),
                    "split": "validation",
                    "loss": validation_losses
                }
            )
        ]
    )


def encode_decode_error_dataframe():
    pass
