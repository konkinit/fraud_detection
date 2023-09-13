<h1 align="center">
    Fraud Detection
    <br/>
</h1>

<p align="center">
    The project consists of implementing an autoencoder-based fraud detector
</p>

<p align="center">
    <img alt="GitHub Workflow Status (with event)" src="https://img.shields.io/github/actions/workflow/status/konkinit/fraud_detection/lint_test.yaml?style=for-the-badge&label=Lint%20%26%20Test%20">
    <br/>
    <img alt="GitHub" src="https://img.shields.io/github/license/konkinit/fraud_detection?style=for-the-badge">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/konkinit/fraud_detection?style=for-the-badge">
    <a href="https://www.python.org/downloads/release/python-3100/" target="_blank">
        <img src="https://img.shields.io/badge/python-3.10-blue.svg?style=for-the-badge" alt="Python Version"/>
    </a>
</p>


## To-Do

- Design an online learning framework | Implement model retraining
- Write Unit Tests
- Search for real data for implementation


## Quick Start

Clone the repo, get in the directory `fraud_detection/` and run the `main.py` program with the args tuned
```bash
git clone https://github.com/konkinit/fraud_detection.git
```
- For training a new model
```bash
python training.py --idmodel 'simulated_data' --rawdatapath './data/simulated_raw_data.gzip' --splitfrac 0.7 0.2 0.1 --codedim 35 --hiddendim 150 --lr 1e-3 --nepochs 50 --mode 'train'
```

- For updating weigths of an existing model
```bash
python training.py --idmodel 'simulated_data' --rawdatapath './data/simulated_raw_data_new_arrival.gzip' --splitfrac 0.7 0.2 0.1 --codedim 35 --hiddendim 150 --lr 1e-3 --nepochs 50 --mode 'retrain'
```

- After training or retraining a model, inference on instances is done by running: 
```bash
uvicorn production:app --reload
```

## References & Citations
