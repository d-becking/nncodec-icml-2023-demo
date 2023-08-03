<div align="center">

![nncodec_logo](https://github.com/d-becking/nncodec-icml-2023-demo/assets/56083075/f310c92e-537e-4960-b223-7ec51c9accc3)

# @ ICML 2023 Neural Compression Workshop

[![Conference](https://img.shields.io/badge/ICML-Paper-blue)](https://openreview.net/forum?id=5VgMDKUgX0)
[![Conference](https://img.shields.io/badge/ICML-Poster-red)](https://datacloud.hhi.fraunhofer.de/s/QFKiqCYE3Q8ptAy)

</div>

Our spotlight paper titled **"NNCodec: An Open Source Software Implementation of the Neural Network Coding 
ISO/IEC Standard"** is presented at the ICML 2023 [Neural Compression Workshop](https://neuralcompression.github.io/workshop23).


**TL;DR** -  The paper presents NNCodec, analyzes its coding tools with respect to the principles of information theory 
and gives comparative results for a broad range of neural network architectures. Find the article [here](https://openreview.net/forum?id=5VgMDKUgX0).

## Table of Contents
- [Information](#information)
- [Installation](#installation)
- [NNCodec Usage](#nncodec-usage):
  * [Reproducibility of paper results](#reproducibility-of-paper-results)
  * [Demo](#demo)
- [Citation and Publications](#citation-and-publications)
- [License](#license)

## Information

This repository is for reproducing the results shown in the paper. Additionally, it includes a Pascal VOC image 
segmentation demo, prepared for the Neural Compression Workshop (ICML'23). How to run the code, reproduce paper results and run the demo is described in the [NNCodec Usage](#nncodec-usage) section.

The official NNCodec git repository, that served as the basis for this repo, can be found here:

[![Conference](https://img.shields.io/badge/fraunhoferhhi-nncodec-green)](https://github.com/fraunhoferhhi/nncodec)

It also contains a [Wiki-Page](https://github.com/fraunhoferhhi/nncodec/wiki) providing further information on NNCodec.

### The Fraunhofer Neural Network Encoder/Decoder (NNCodec)
The Fraunhofer Neural Network Encoder/Decoder Software (NNCodec) is an efficient implementation of NNC ([Neural Network Coding ISO/IEC 15938-17](https://www.iso.org/standard/78480.html)), which is the first international standard on compression of neural networks.
NNCodec provides an encoder and decoder with the following main features:
- Standard compliant implementation of the core compression technologies including, e.g., DeepCABAC
- Easy-to-use user interface
- Built-in support for common frameworks (e.g. TensorFlow and PyTorch)
- Built-in ImageNet-support for data driven compression tools
- Extensibility of data driven compression tools for arbitrary datasets

## Installation

The software provides python packages which can be installed using pip. However, core technologies are implemented using C++, which requires a C++ compiler for the installation process.

The software has been tested on different target platforms (Windows, Linux and MacOS).

### Requirements

- python >= 3.6 (recommended versions 3.6, 3.7 and 3.8) with working pip
- **Windows:** Microsoft Visual Studio 2015 Update 3 or later

### Package installation

- For **_Windows_**, it is recommended to issue from the root of the cloned repository

```
pip install wheel
pip install -r requirements_cu11.txt
pip install .
```

and for CUDA10 support replace 'requirements_cu11.txt' by 'requirements.txt'.

- On **_Linux/Mac_**, the scripts `create_env.sh` and `create_env_cu11.sh` (for CUDA 11 support) set up a virtual python environment "env" and install all required packages and the software itself, automatically. For activating this environment, issue:

```
source env/bin/activate
```

**Note**: For further information on how to set up a virtual python environment (also on **Windows**) refer to https://docs.python.org/3/library/venv.html .

When successfully installed, the software outputs the line : "Successfully installed NNC-0.2.2"

### Importing the main module

After installation the software can be used by importing the main python module 'nnc':
```python
import nnc
```

## NNCodec Usage

Before running the code, first create the environment as described in [Installation](#installation), and activate it.

### Reproducibility of paper results
Execute 
```shell
python main.py --help
```
for parser argument descriptions.

`--dataset_path` must be specified in accordance with your local data directories. For the CIFAR experiment, the data will be downloaded (< 200MB) to --dataset_path if the data is not available there.

The `--verbose` flag enables extended stdout printing of intermediate results and progress of the compression pipeline.

**Quantization:**

The `--qp` argument specifies an integer quantization step size for weight matrices (usually in the range of -20 to -50, where -50 indicates finer quantization and -20 indicates coarser quantization). The `--nonweight_qp` argument sets the qp for 1D parameter types and non-trainable parameters like BatchNorm statistics (defaults to --nonweight_qp=-75).

#### CIFAR-100 experiments (w/ ResNet56):

Basic compression setting w/o advanced coding tools and a quantization parameter (qp) of -20:

```shell
python main.py --model=resnet56 --model_path=./example/ResNet56_CIF100.pt --dataset=CIFAR100dataset --dataset_path=<YOUR_PATH> --qp=-20
```

For applying dependent quantization (DQ), qp-optimization and BatchNorm folding (BNF) add 

```shell
--use_dq --opt_qp --bnf
```

Applying local scaling adaptation (LSA) additionally requires a learning rate (lr) and a number of training epochs. We used an initial lr of 1e-3 and 30 training epochs for our experiments (and a --batch_size=64):

```shell
--lsa --lr=1e-3 --epochs=30
```
#### ImageNet experiments:

For ImageNet experiments with ResNet50, replace --dataset and --model by 
```shell
--dataset=ImageNet --model=resnet50
```
EfficientNet and Vision Transformer deployment is achieved by setting --model to "efficientnet_b0" and "vit_b_16", respectively. In general, all models of the torchvision model zoo can be inserted here (see output of `python main.py --help`). `--model_path` is not required, if a pretrained model should be downloaded from the torchvision model zoo.

#### Logging (comparative) results using Weights & Biases

We used Weights & Biases (wandb) for experiment logging. Enabling `--wandb` also enables Huffman and bzip2 encoding of the data payloads and the calculation of the Shannon entropy. If you want to use it, add your wandb key and optionally an experiment identifier for the run (--wandb_run_name).

```shell
--wandb, --wandb_key, --wandb_run_name
```

### Demo

The image segmentation demo (using Pascal VOC data) is more visual and demonstrates model degradation due to compression in the form of inaccurate segmentation masks, compared to potentially less expressive top-1 accuracies in image classification:

<img src="https://github.com/d-becking/nncodec-icml-2023-demo/assets/56083075/b721654a-a2b3-4493-9828-a18f79bc0451"  width="500">

To test different quantization parameters and coding tools, the jupyter notebook `icml_demo.ipynb` can be used. Alternatively, use the `main.py` document with the following parser arguments:

```shell
python main.py --model=deeplabv3_resnet50 --dataset=VOC --dataset_path=./example/VOC_demo --plot_segmentation_masks --batch_size=1 --qp=-44 --verbose
```

Any of the Pascal VOC image sampes can be used for the demo. Just update/replace the samples, ground truth and *.txt files in ./example/VOC_demo/VOCdevkit/VOC2012/.

## Citation and Publications
If you use NNCodec in your work, please cite:
```
@inproceedings{becking2023nncodec,
title={{NNC}odec: An Open Source Software Implementation of the Neural Network Coding {ISO}/{IEC} Standard},
author={Daniel Becking and Paul Haase and Heiner Kirchhoffer and Karsten M{\"u}ller and Wojciech Samek and Detlev Marpe},
booktitle={ICML 2023 Workshop Neural Compression: From Information Theory to Applications},
year={2023},
url={https://openreview.net/forum?id=5VgMDKUgX0}
}
```
### Publications
- D. Becking, et al. **"NNCodec: An Open Source Software Implementation of the Neural Network Coding ISO/IEC Standard"**, 40th International Conference on Machine Learning (ICML), 2023, Neural Compression Workshop (Spotlight)
- H. Kirchhoffer, et al. **"Overview of the Neural Network Compression and Representation (NNR) Standard"**, IEEE Transactions on Circuits and Systems for Video Technology, pp. 1-14, July 2021, doi: 10.1109/TCSVT.2021.3095970, Open Access
- P. Haase et al., **"Encoder Optimizations For The NNR Standard On Neural Network Compression"**, 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 3522-3526, doi: 10.1109/ICIP42928.2021.9506655.
- K. Müller et al., **"Ein internationaler KI-Standard zur Kompression Neuronaler Netze"**, FKT- Fachzeitschrift für Fernsehen, Film und Elektronische Medien, pp. 33-36, September 2021
- S. Wiedemann et al., **"DeepCABAC: A universal compression algorithm for deep neural networks"**, in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2020.2969554.

## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of the use of the contents of this repository.

For more information and bug reports, please contact: nncodec@hhi.fraunhofer.de

**Copyright (c) 2019-2023, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.**

**All rights reserved.**
