<<<<<<< HEAD
# Commands to train and evaluate
## Mnist
python main.py \
--mode train \
--dataset mnist \
-lr 0.001 \
--epochs 50 \
--verbose

python main.py \
--mode evaluate \
--dataset mnist \
--pretrain-path ../logs/ckpt/checkpoint_lenet_mnist.keras

## FMnist
python main.py \
--mode train \
--dataset fmnist \
-lr 0.001 \
--epochs 50 \
--verbose

python main.py \
--mode evaluate \
--dataset fmnist \
--pretrain-path ../logs/ckpt/checkpoint_lenet_fmnist.keras

## Caltech101
python main.py \
--mode train \
--dataset caltech101 \
--raw-dir ... \
-lr 0.001 \
--epochs 50 \
--verbose

python main.py \
--mode evaluate \
--dataset caltech101 \
--raw-dir ... \
--pretrain-path ../logs/ckpt/checkpoint_lenet_caltech101.keras

## Caltech256
python main.py \
--mode train \
--dataset caltech256 \
--raw-dir ... \
-lr 0.001 \
--epochs 50 \
--verbose

python main.py \
--mode evaluate \
--dataset caltech256 \
--raw-dir ... \
--pretrain-path ../logs/ckpt/checkpoint_lenet_caltech256.keras

# Commands to finetune
## Caltech101
python main.py \
--mode fine-tune \
--dataset caltech101 \ 
--raw-dir ... \
-lr 0.0001 \
--epochs 50 \
--verbose \
--pretrain-path ../logs/ckpt/checkpoint_lenet_mnist.keras

python main.py \
--mode fine-tune \
--dataset caltech101 \ 
--raw-dir ... \
-lr 0.0001 \
--epochs 50 \
--verbose \
--pretrain-path ../logs/ckpt/checkpoint_lenet_fmnist.keras

## Caltech256
python main.py \
--mode fine-tune \
--dataset caltech256 \ 
--raw-dir ... \
-lr 0.0001 \
--epochs 50 \
--verbose \
--pretrain-path ../logs/ckpt/checkpoint_lenet_mnist.keras

python main.py \
--mode fine-tune \
--dataset caltech256 \ 
--raw-dir ... \
-lr 0.0001 \
--epochs 50 \
--verbose \
--pretrain-path ../logs/ckpt/checkpoint_lenet_fmnist.keras


## Evaluate
python main.py \
--mode evaluate \
--dataset caltech256 \
--raw-dir ... \
--pretrain-path ...

python main.py \
--mode evaluate \
--dataset caltech256 \
--raw-dir ... \
--pretrain-path ...
=======
## Training
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

## Evaluating
```Console
python main.py \
--mode evaluate \
--dataset dataset-name \
--pretrain-path path/to/pretrain/model
```
* For Caltech 101 and 256, add --raw-dir path/to/dataset

## Fine-tuning
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
>>>>>>> e8aff61cbd56e4e29b67570098a35c1b3b1285a0
