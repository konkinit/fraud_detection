import os
import sys
from argparse import ArgumentParser
import plotly.express as px

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Model_Trainer
from src.conf import PLOTTING_FEATURES


parser = ArgumentParser(
    description="Model training args parser"
)
parser.add_argument(
    '--idmodel', type=str,
    help="""
    Model unique identifier""", required=True
)
parser.add_argument(
    '--rawdatapath', type=str,
    help="""
    Raw data storage path""", required=True
)
parser.add_argument(
    '--splitfrac', nargs=3, type=float,
    help="""
    Split fractions provided in the order train,
    validation and test proportions""", required=True
)
parser.add_argument(
    "--hiddendim", type=int,
    help="Encoder hidden layer dimension", required=True
)
parser.add_argument(
    "--codedim", type=int,
    help="Auto-encoder code layer dimension", required=True
)
parser.add_argument(
    "--lr", type=float,
    help="Learning rate", required=True
)
parser.add_argument(
    "--nepochs", type=int,
    help="Number of epochs", required=True
)
parser.add_argument(
    "--mode", type=str,
    help="Mode of training either train a new model or update \
    an existing model weigths", required=True
)
args = parser.parse_args()


if __name__ == "__main__":

    _model_trainer = Model_Trainer(
        model_id=args.idmodel,
        raw_data_path=args.rawdatapath,
        data_split_fractions=args.splitfrac,
        hidden_dim=args.hiddendim,
        code_dim=args.codedim,
        learning_rate=args.lr,
        n_epochs=args.nepochs
    )

    _model_trainer.update_weights(mode=args.mode)

    _model_trainer.encode_decode_error()

    _model_trainer.save_metadata()

    _fig = px.histogram(
        _model_trainer.encode_decode_errors_dataframe,
        x=PLOTTING_FEATURES.RECONSTRUCTION_ERROR,
        color=PLOTTING_FEATURES.COLOR
    )
    fig = px.line(
        _model_trainer.losses_dataframe,
        x=PLOTTING_FEATURES.X,
        y=PLOTTING_FEATURES.Y,
        color=PLOTTING_FEATURES.COLOR,
    )

    _fig.write_image(
        "./data/figs/reconstruction_error_X_split.png", engine="kaleido",
    )
    fig.write_image(
        "./data/figs/losses_X_epoch.png", engine="kaleido",
    )
