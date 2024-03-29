# Build: docker build -t deepmc-env .
# Run: docker run --shm-size 50G -v $(pwd):/DeepMC --gpus all -it --rm --name deepmc-dev deepmc-env 

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates build-essential cmake curl wget git \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& chmod +x ~/miniconda.sh \
	&& ~/miniconda.sh -b -p /opt/conda \
	&& rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install -y python=3.8 \
	&& conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia \
	&& conda install -y jupyter matplotlib \
	&& conda clean -ya

RUN python -m pip install pytorch-lightning==1.6.5 torchmetrics hydra-core hydra-colorlog \
	hydra-optuna-sweeper wandb neptune-client mlflow comet-ml pyrootutils \
	pre-commit rich pytest sh pyquaternion

SHELL ["/bin/bash", "-c"]

WORKDIR /
