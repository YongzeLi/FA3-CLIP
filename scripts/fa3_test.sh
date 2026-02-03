#!/bin/bash
# custom config
# val checkpoint path:
# test/
# └── p1/
#     └── CLIP@VL/
#         └── model-best.pth.tar

DATA=/UniAttackData  # Dataset dir
DATASET=UniAttackData
MODELDIR=./test/p1  # load pth dir
OUTPUT=./test/  # pth save pth
PROTOCOL=p1@train@val@test # {protocol name}@train@val@test
GPU_IDS='0' # str
TRAINER=FA3  # backbone
VERSION=VL # V or VL
PROMPT=class # class, engineering, ensembling
CFG=vit_amp # config file
PROMPTLENGTH=6 # prompt length
LR='0.000001'

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}@${PROMPT}/${CFG}/${PROTOCOL}/seed${SEED}/
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --gpu_ids ${GPU_IDS} \
    --root ${DATA} \
    --protocol ${PROTOCOL} \
    --seed ${SEED} \
    --LR ${LR} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODELDIR} \
    --version ${VERSION} \
    --prompt ${PROMPT} \
    --prompt_length ${PROMPTLENGTH} \
    --no-train \
    --eval-only \
    # fi
done