from pandas import read_parquet, Dataframe
import s3fs
import yaml


def read_data_from_s3(
        file: str,
        access_keys: dict = None
) -> Dataframe:
    if access_keys is None:
        with open('./data/acces-key.yaml', 'r') as file:
            access_keys = yaml.safe_load(file)

    fs = s3fs.S3FileSystem(
        key=access_keys["key"],
        secret=access_keys["secret"]
    )

    return read_parquet(
        fs.open(
            fs.ls(
                f'customer-data-platform-retail/test_data/{file}'
            )
        )
    )
