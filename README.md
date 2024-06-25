# LeNet Experiment 🧪

Welcome to the LeNet Experiment repository! This project contains an implementation of the LeNet-5 Convolutional Neural Network (CNN) model, originally designed for handwritten digit classification. The goal of this project is to experiment with and understand the fundamental concepts of CNNs by using the LeNet-5 architecture.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

LeNet-5 is one of the earliest convolutional neural network architectures, proposed by Yann LeCun et al. in their 1998 paper. It was primarily used for character recognition tasks such as reading zip codes, digits, etc.

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```sh
git clone https://github.com/phanhoang1803/LeNet_Experiment.git
cd LeNet_Experiment
pip install -r requirements.txt
```

## Usage

### Training
* To train
```Console
  python main.py \
  --mode train \
  --dataset dataset-name \
  --raw-dir ... \
  -lr 0.001 \
  --epochs 50 \
  --verbose
```
* For Caltech 101 and 256, add --raw-dir path/to/dataset

### Evaluating
```Console
python main.py \
--mode evaluate \
--dataset dataset-name \
--pretrain-path path/to/pretrain/model
```
* For Caltech 101 and 256, add --raw-dir path/to/dataset

### Fine-tuning
```Console
python main.py \
--mode fine-tune \
--dataset dataset-name \
--raw-dir path/to/dataset \
-lr 0.0001 \
--epochs 50 \
--verbose \
--pretrain-path path/to/pretrain/model 
```

## Architecture

The LeNet-5 architecture consists of the following layers:

1. **Input Layer:** 32x32 grayscale image.
2. **C1 Convolutional Layer:** 6 feature maps, kernel size 5x5, followed by a tanh activation function.
3. **S2 Subsampling Layer:** 6 feature maps, 2x2 kernel, and a stride of 2 (average pooling).
4. **C3 Convolutional Layer:** 16 feature maps, 5x5 kernel, followed by a tanh activation function.
5. **S4 Subsampling Layer:** 16 feature maps, 2x2 kernel, and a stride of 2 (average pooling).
6. **C5 Convolutional Layer:** 120 feature maps, 5x5 kernel, followed by a tanh activation function.
7. **F6 Fully Connected Layer:** 84 units, followed by a tanh activation function.
8. **Output Layer:** 10 units (for digit classification), followed by a softmax activation function.

## Results

The performance of the model on various datasets is shown in the table below:

| Model                 | MNIST      | FMNIST     | Caltech101 | Caltech256 |
|-----------------------|------------|------------|------------|------------|
| **CNN**               | 0.9865     | 0.8731     | 0.3745     | 0.0257     |
| **CNN-MNIST Finetune**| -          | -          | 0.3190     | 0.0981     |
| **CNN-FMNIST Finetune**| -          | -          | 0.3577     | 0.1250     |
| **ANN**               | 0.9832     | 0.8867     | 0.4006     | 0.1174     |
| **ANN-MNIST Finetune**| -          | -          | 0.3316     | 0.1039     |
| **ANN-FMNIST Finetune**| -          | -          | 0.3724     | 0.1298     |

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request. Please ensure your changes are well-documented and tested.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Yann LeCun and his colleagues for developing the LeNet-5 architecture.
- The open-source community for providing valuable resources and inspiration.
- [Phan Hoang](https://github.com/phanhoang1803) for maintaining this repository.
