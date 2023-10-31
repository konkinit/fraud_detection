<h1 align="center">
    Fraud Detection
    <br/>
</h1>

<p align="center">
    The project consists of implementing an autoencoder-based fraud detector 
    on customers' data
</p>

<p align="center">
    <img alt="GitHub Workflow Status (with event)" src="https://img.shields.io/github/actions/workflow/status/konkinit/fraud_detection/cicd_workflow.yaml?style=for-the-badge&label=Lint%2C%20Test%20%26%20Build%20Docker%20Image">
    </br>
    <img alt="GitHub" src="https://img.shields.io/github/license/konkinit/fraud_detection?style=for-the-badge">
    <a href="https://www.python.org/downloads/release/python-3100/" target="_blank">
        <img src="https://img.shields.io/badge/python-3.10-blue.svg?style=for-the-badge" alt="Python Version"/>
    </a>
    </br>
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/konkinit/fraud_detection?style=for-the-badge">
    <img alt="Docker Image Size (tag)" src="https://img.shields.io/docker/image-size/kidrissa/fraud_detector_app/latest?style=for-the-badge&label=Image%20Size">
</p>


## To-Do

- Design an online learning framework | Implement model retraining
- Search for real data for implementation


## Quick Start

- Clone the repo, get in the directory `fraud_detection/` 
```bash
git clone https://github.com/konkinit/fraud_detection.git

cd ./fraud_detection

pip install -r requirements.txt
```

- For training a new model, run the following command with the tuned args
```bash
python training.py --help
```
```bash
python training.py --mode 'train' --idmodel 'simulated_data' --rawdatapath 's3_data_raw.gzip' --splitfrac 0.7 0.2 0.1 --codedim 35 --hiddendim 150 --lr 1e-3 --nepochs 50
```

- For updating weigths of an existing model (ensure the dimensions passed through the args are 
the same as the current model dimensions)
```bash
python training.py --mode 'retrain' --idmodel 'simulated_data' --rawdatapath 's3_data_raw_new_arrival.gzip' --splitfrac 0.7 0.2 0.1 --codedim 35 --hiddendim 150 --lr 1e-3 --nepochs 50
```

- After training or retraining a model, inference on instances is done by running: 
```bash
uvicorn production:app --port 8800 --reload
```
The endpoint looks like `/customer_id/{customer}?model={model_id}` where `{customer}` 
refers to an identifier of a customer and `{model_id}` is the deployed fraud detector model.

## Description

One phenomenon businesses face undoubtedly is fraud. It is a situation where
a customer has an irregular pattern of events (transactions, visits, ...) with a business. 
Two factions of customers emerge : the atypical or frauder and the typical customers. It is 
important to notice that fraud is rare event that is to say in a sample of 1000 customers, 
up to 5 appear to have a fraudulent behaviours. Gather, in a customer base, a large number 
of typical customers is then realistic conequently train a model aiming to identify regular 
behaviours and reconstruct a typical customer profile is possible. It turns out that 
AutoEncoders perform this task.

## Model



## References & Citations
