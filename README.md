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
--pretrain-path ../logs/ckpt/checkpoint_lenet_mnist.keras
```
