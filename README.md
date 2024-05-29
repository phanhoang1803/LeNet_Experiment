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
