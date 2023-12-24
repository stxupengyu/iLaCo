#!/bin/bash

DATASET=$1
RATE=$2
GPUID=$3

if [ "$DATASET" = "eurlex" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 50 --batch 128  \
    --alpha 0.7 --beta 0.001 \
    --rho $RATE

elif [ "$DATASET" = "rcv" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 50 --batch 256  \
    --alpha 0.7 --beta 0.05 \
    --rho $RATE

elif [ "$DATASET" = "aapd" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 50 --batch 256  \
    --alpha 0.7 --beta 0.05 \
    --rho $RATE
fi
