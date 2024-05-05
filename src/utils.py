import os
import sys
import s3fs
import awswrangler
import boto3
from dotenv import load_dotenv
# from functools import lru_cache
from numpy import array, append, random
from torch import device, cuda, load
from typing import List, Any
from sklearn.metrics import recall_score, confusion_matrix, precision_score
from pandas import read_parquet, DataFrame, concat


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


load_dotenv()


def upload_model_in_s3(
        _local_file: str,
) -> None:

    boto3_session = boto3.Session(
        aws_access_key_id=os.getenv("S3_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET"),
        region_name=os.getenv("S3_REGION")
    )

    awswrangler.s3.upload(
        local_file=f"./models/{_local_file}",
        path=f"s3://models-factory/{_local_file}",
        boto3_session=boto3_session
    )


def read_model_from_s3(model_id: str, device_id: int = 1) -> Any:
    device = get_device(device_id)

    fs = s3fs.S3FileSystem(
        key=os.getenv("S3_KEY"),
        secret=os.getenv("S3_SECRET")
    )

    model_cktp = fs.open(
        fs.ls(
            str(f"models-factory/best_model_{model_id}.ckpt")
        )[0]
    )

    return load(model_cktp, map_location=device)


def upload_data_in_s3(file_name: str, folder: str = 'data') -> None:

    boto3_session = boto3.Session(
        aws_access_key_id=os.getenv("S3_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET"),
        region_name=os.getenv("S3_REGION")
    )

    awswrangler.s3.upload(
        local_file=f"./data/{file_name}",
        path=f"s3://customer-data-platform-retail/{folder}/{file_name}",
        boto3_session=boto3_session
    )


def read_data_from_s3(
        file_name: str
) -> DataFrame:

    fs = s3fs.S3FileSystem(
        key=os.getenv("S3_KEY"),
        secret=os.getenv("S3_SECRET")
    )

    return read_parquet(
        fs.open(
            fs.ls(
                str(
                    f"customer-data-platform-retail/{file_name}"
                )
            )[0]
        )
    )


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


def cutoff_evaluation() -> float:
    ...
