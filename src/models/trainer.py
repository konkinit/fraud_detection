import os
import sys
import torch
from pandas import read_parquet
from typing import (
    List
)

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
            n_epochs: int
    ) -> None:
        self.model_id = model_id
        # Data setting
        raw_data = read_parquet(raw_data_path).values
        self.raw_data = torch.from_numpy(raw_data).to(DEVICE)
        (
            self.train_dataloader,
            self.validation_dataloader,
            self.test_dataloader
        ) = DataLoaders(
            self.raw_data, data_split_fractions
        ).get_dataloaders()

        self.layers_dim = LAYERS_DIMS(
            INPUT_DIM=self.raw_data.shape[1],
            HIDDEN_DIM=hidden_dim,
            CODE_DIM=code_dim
        )
        self.model_hyperparams = MODEL_FEATURES(
            LEARNING_RATE=learning_rate,
            N_EPOCHS=n_epochs
        )

    def train_config(self, **kwargs) -> None:
        """Model config for first of all training
        """
        self.model = FraudAutoEncoder(self.layers_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_hyperparams.LEARNING_RATE
        )
        self.loss_criterion = torch.nn.MSELoss()

    def retrain_config(self, **kwargs) -> None:
        """Model configuration for retraining
        """
        checkpoint = torch.load(
            "./models/best_model_simulated_data.ckpt", map_location="cuda:0"
        )
        self.train_config()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

        self.model.eval()

    def update_weights(self, mode: str, **kwargs) -> None:
        """Train the model on training data and make inference
        """
        assert mode in ("train", "retrain")
        self.train_config() if mode == "train" else self.retrain_config()
        training_losses, validation_losses = [], []
        minimum_validation_loss = float("inf")
        best_epoch = None
        best_model_state_dict = None
        best_optimizer_state_dict = None
        for epoch in range(self.model_hyperparams.N_EPOCHS):
            # Weights update for each epoch
            epoch_train_loss = 0.0
            for _, inputs in enumerate(self.train_dataloader):
                # Zero gradients for every batch!
                self.optimizer.zero_grad()

                # Make encoding and decoding
                reconstitued_inputs = self.model(inputs)

                # Compute loss and its gradients on trining datasets
                training_loss = self.loss_criterion(
                    inputs, reconstitued_inputs
                )
                training_loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Evaluate epoch loss
                epoch_train_loss += float(
                    training_loss.data) / len(self.train_dataloader)

            training_losses.append(epoch_train_loss)

            # Inference after each epoch on validation data
            epoch_validation_loss = 0.0
            for _, validation_input in enumerate(self.validation_dataloader):
                reconstitued_validation_input = self.model(validation_input)
                with torch.no_grad():
                    validation_loss = self.loss_criterion(
                        validation_input, reconstitued_validation_input
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

        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state_dict,
            'optimizer_state_dict': best_optimizer_state_dict
        }, f"./models/best_model_{self.model_id}.ckpt")

        self.losses_dataframe = losses_dataframe(
            self.model_hyperparams.N_EPOCHS, training_losses, validation_losses
        )
