<div align="center">

# DeepMC

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Marker based motion capture solving. This repository contains Pytorch Lightning implementation of [MoCap-Solver](https://github.com/NetEase-GameAI/MoCap-Solver), as well as pose estimation models for [Vector-Neurons](https://github.com/FlyingGiraffe/vnn)

The code follows the structure of [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)

## Installation

Easiest way to reproduce the exact environment is to use Docker and build the image using following command:
```
docker build -t deepmc-env .
```

Dependencies can be checked from [Dockerfile](https://github.com/beyaldiz/DeepMC/blob/main/Dockerfile).

## Usage

### Data Preparation

For the experiments, dataset provided by [MoCap-Solver](https://github.com/NetEase-GameAI/MoCap-Solver) is used. After synthesizing the dataset copy it under `data/` folder.

### Data Preprocessing

Data needs to be preprocessed and saved for faster loading during training. The preprocessing is done automatically if the preprocessed files are not found, for more details check the dataloaders [encoders_dataset.py](https://github.com/beyaldiz/DeepMC/blob/main/deepmc/datamodules/components/MoCap_Solver/encoders_dataset.py).

For different normalizations of the data, tools in [create_data.py](https://github.com/beyaldiz/DeepMC/blob/main/deepmc/utils/MoCap_Solver/create_data.py) can be used. `msalign: Alignment heuristic provided by MoCap-Solver`, `gtalign: Alignment based on root joint`, `noalign: No normalization`.


