FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Define args
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set user
USER ${USERNAME}

ENV PATH="/home/${USERNAME}}/.local/bin:${PATH}"

COPY --chown=${USERNAME}:${USERNAME} . /home/${USERNAME}/fraud_detection

WORKDIR /home/${USERNAME}/fraud_detection

RUN pip install -r requirements.txt --user

EXPOSE 8800

ENTRYPOINT ["python", "-m", "uvicorn", "production:app", "--port", "8800", "--reload"]
