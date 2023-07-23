import os
import sys
import torch
import numpy as np
import plotly.express as px

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import LAYERS_DIMS, MODEL_FEATURES
from src.models import FraudAutoEncoder
from src.data import DataLoaders
from src.utils import get_device, losses_dataframe


DEVICE = get_device(1)


if __name__ == "__main__":
    # Data prep
    SAMPLING_PARAMS = {"low": -1, "high": 1, "size": [5_000, 300]}
    DATA_SPLIT_FRACS = [0.7, 0.2, 0.1]
    simul_data = np.random.uniform(**SAMPLING_PARAMS)
    _simul_data = torch.from_numpy(simul_data).to(DEVICE)

    # Model configuration
    INPUT_DIM = _simul_data.shape[1]
    _LAYERS_DIMS = LAYERS_DIMS(
        ENCODER_HIDDEN_DIM=150, ENCODER_OUTPUT_DIM=16, DECODER_HIDDEN_DIM=150
    )
    model = FraudAutoEncoder(INPUT_DIM, LAYERS_DIMS).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_FEATURES.LEARNING_RATE
    )
    loss_criterion = torch.nn.MSELoss()

    # Get Pytorch Dataloaders
    (train_dataloader, validation_dataloader, test_dataloader) = DataLoaders(
        _simul_data, DATA_SPLIT_FRACS
    ).get_dataloaders()

    # Model Training
    training_losses, validation_losses = [], []

    for epoch in range(MODEL_FEATURES.N_EPOCHS):
        # Weights update for each epoch
        epoch_train_loss = 0.0
        for _, inputs in enumerate(train_dataloader):
            # Zero gradients for every batch!
            optimizer.zero_grad()

            # Make encoding and decoding
            reconstitued_inputs = model(inputs)

            # Compute loss and its gradients on trining datasets
            training_loss = loss_criterion(inputs, reconstitued_inputs)
            training_loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Evaluate epoch loss
            epoch_train_loss += float(
                training_loss.data) / len(train_dataloader)

        training_losses.append(epoch_train_loss)

        # Inference after each epoch on validation data
        epoch_validation_loss = 0.0
        for _, validation_inputs in enumerate(validation_dataloader):
            reconstitued_validation_inputs = model(validation_inputs)
            with torch.no_grad():
                validation_loss = loss_criterion(
                    validation_inputs, reconstitued_validation_inputs
                )
                epoch_validation_loss += float(validation_loss.data) / len(
                    validation_dataloader
                )

        validation_losses.append(epoch_validation_loss)

        print(
            """
            Epoch {:} : Training loss = {:.6f} | Validation loss = {:.6f}
            """.format(
                epoch, training_loss, validation_loss
            )
        )

    # Inference
    df_losses = losses_dataframe(
        MODEL_FEATURES.N_EPOCHS, training_losses, validation_losses
    )

    fig = px.line(
        df_losses, x="epoch", y="loss", color="split", height=500, width=800
    )
    fig.show()
