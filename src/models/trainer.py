import os
import sys
import torch
from datetime import date
from pandas import read_parquet, DataFrame, concat
from tinydb import TinyDB
from typing import List

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import LAYERS_DIMS, MODEL_FEATURES
from src.models import FraudAutoEncoder
from src.data import DataLoaders
from src.utils import get_device, losses_dataframe


DEVICE = get_device(1)


class Model_Trainer:
    def __init__(
        self,
        model_id: str,
        raw_data_path: str,
        data_split_fractions: List[float],
        hidden_dim: int,
        code_dim: int,
        learning_rate: float,
        n_epochs: int,
    ) -> None:
        self.model_id = model_id
        # Data setting
        raw_data = read_parquet(raw_data_path)
        (
            self.train_dataloader,
            self.validation_dataloader,
            self.test_dataloader,
        ) = DataLoaders(raw_data, data_split_fractions).get_dataloaders()

        self.layers_dim = LAYERS_DIMS(
            INPUT_DIM=raw_data.shape[1], HIDDEN_DIM=hidden_dim,
            CODE_DIM=code_dim
        )
        self.model_hyperparams = MODEL_FEATURES(
            LEARNING_RATE=learning_rate, N_EPOCHS=n_epochs
        )
        self.model_params = {
            "input_dim": raw_data.shape[1],
            "hidden_dim": hidden_dim,
            "code_dim": code_dim
        }

    def train_config(self, **kwargs) -> None:
        """Model config for first of all training"""
        self.model = FraudAutoEncoder(self.layers_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_hyperparams.LEARNING_RATE
        )
        self.loss_criterion = torch.nn.MSELoss()

    def retrain_config(self, **kwargs) -> None:
        """Model configuration for retraining"""
        model_path = f"./models/best_model_{self.model_id}.ckpt"
        if os.path.isfile(os.path.join(model_path)):
            checkpoint = torch.load(
                "./models/best_model_simulated_data.ckpt",
                map_location="cuda:0"
            )
            self.train_config()
            # assert indifference on params
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]

            self.model.eval()
        else:
            pass

    def update_weights(self, mode: str, **kwargs) -> None:
        """Train the model on training data and make inference"""
        assert mode in ("train", "retrain")
        self.mode = mode
        self.train_config() if mode == "train" else self.retrain_config()
        training_losses, validation_losses = [], []
        minimum_validation_loss = float("inf")
        best_epoch = None
        best_model_state_dict = None
        best_optimizer_state_dict = None
        for epoch in range(self.model_hyperparams.N_EPOCHS):
            # Weights update for each epoch
            epoch_train_loss = 0.0
            for _, (batch_ids, batch_samples) in enumerate(
                    self.train_dataloader
            ):
                # Zero gradients for every batch!
                self.optimizer.zero_grad()

                # Make encoding and decoding
                reconstitued_batch_samples = self.model(batch_samples)

                # Compute loss and its gradients on trining datasets
                training_loss = self.loss_criterion(
                    batch_samples, reconstitued_batch_samples
                )
                training_loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Evaluate epoch loss
                epoch_train_loss += float(training_loss.data) / len(
                    self.train_dataloader
                )

            training_losses.append(epoch_train_loss)

            # Inference after each epoch on validation data
            epoch_validation_loss = 0.0
            for _, (batch_ids, batch_samples) in enumerate(
                    self.validation_dataloader
            ):
                reconstitued_batch_samples = self.model(batch_samples)
                with torch.no_grad():
                    validation_loss = self.loss_criterion(
                        batch_samples, reconstitued_batch_samples
                    )
                    epoch_validation_loss += float(validation_loss.data) / len(
                        self.validation_dataloader
                    )

            if epoch_validation_loss < minimum_validation_loss:
                minimum_validation_loss = epoch_validation_loss
                best_epoch = epoch
                best_model_state_dict = self.model.state_dict()
                best_optimizer_state_dict = self.optimizer.state_dict()

            validation_losses.append(epoch_validation_loss)

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_model_state_dict,
                "optimizer_state_dict": best_optimizer_state_dict,
            },
            f"./models/best_model_{self.model_id}.ckpt",
        )

        self.losses_dataframe = losses_dataframe(
            self.model_hyperparams.N_EPOCHS, training_losses, validation_losses
        )

    def encode_decode_error(self) -> None:
        """Evaluate encode-decode error
        """
        self.retrain_config()
        error_func = torch.nn.MSELoss()
        self.encode_decode_errors_dataframe: DataFrame = DataFrame(
            columns=["id", "encode_decode_error", "split"]
        )
        dataloader_split_dict = {
            "train": self.train_dataloader,
            "validation": self.validation_dataloader,
            "test": self.test_dataloader
        }
        for split in list(dataloader_split_dict.keys()):
            for _, (batch_ids, batch_samples) in enumerate(
                    dataloader_split_dict[split]
            ):
                assert batch_ids.shape[0] == batch_samples.shape[0]
                for i in range(batch_ids.shape[0]):
                    with torch.no_grad():
                        _x = batch_samples[i]
                        _error = error_func(_x, self.model(_x))
                        self.encode_decode_errors_dataframe = concat(
                            [
                                self.encode_decode_errors_dataframe,
                                DataFrame(
                                    data=[
                                        {
                                            "id": batch_ids[i].data,
                                            "reconstruction_error": float(
                                                _error.data),
                                            "split": split,
                                        }
                                    ]
                                ),
                            ]
                        )

        self.fraud_cutoff = self.encode_decode_errors_dataframe[
            "reconstruction_error"].max()

    def save_metadata(self) -> None:
        """Store model training metadata
        """
        db = TinyDB('./data/models_metadata/metadata_history.json')
        table = db.table('metadata_history')
        table.insert(
            {
                "model_id": self.model_id,
                **self.model_params,
                "updating_date": date.today().strftime(
                    "%Y-%m-%d"
                ),
                "action": self.mode,
                "fraud_cutoff": self.fraud_cutoff
            }
        )
