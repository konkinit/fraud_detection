FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

COPY . /fraud_detection

WORKDIR /home/${USERNAME}/fraud_detection

RUN pip install -r requirements.txt

EXPOSE 8800

ENTRYPOINT ["python", "-m", "uvicorn", "production:app", "--port", "8800", "--reload"]
