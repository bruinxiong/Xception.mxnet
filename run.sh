#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --batch-size, --gpus.

## train xception for imagenet
python -u train_xception.py --data-dir data/imagenet --data-type imagenet --batch-size 256 --gpus=0,1,2,3
python -u train_xception.py --data-dir data/imagenet --data-type imagenet --batch-size 256 --gpus=0,1,2,3 --model-load-epoch=50 --lr 0.001 --retrain

