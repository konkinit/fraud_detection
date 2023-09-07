import os
import sys
from argparse import ArgumentParser
import numpy as np
import plotly.express as px

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Model_Trainer


parser = ArgumentParser(
    description="Model training args parser"
)
parser.add_argument(
    '--splitfrac', nargs=3, type=float,
    help="""
    Split fractions provided in the order train,
    validation and test proportions
    """, required=True
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
args = parser.parse_args()


if __name__ == "__main__":
    sampling_params = {"low": -1, "high": 1, "size": [5_000, 300]}
    simul_data = np.random.uniform(**sampling_params)

    _model_trainer = Model_Trainer(
        model_id="sampled_data",
        raw_data=simul_data,
        data_split_fractions=args.splitfrac,
        hidden_dim=args.hiddendim,
        code_dim=args.codedim,
        learning_rate=args.lr,
        n_epochs=args.nepochs
    )

    _model_trainer.train()

    fig = px.line(
        _model_trainer.losses_dataframe, x="epoch", y="loss",
        color="split", height=500, width=800
    )
    fig.show()
