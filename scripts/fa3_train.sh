#!/bin/bash
# custom config
# Dataset path:
# UniAttackData/
# └── Data/
# └── Protocol/
#     └── P1/
#         └── train.txt
#         └── val.txt
#         └── test.txt
#     └── P2.1/
#         └── ...

GPU_IDS='0'
LR='0.000001'
DATASET=UniAttackData
DATA=/UniAttackData
PROTOCOL=p1@train@val@test
PREPROCESS=resize_crop_rotate_flip   ### resize_crop_rotate_flip_ColorJitter
OUTPUT=./test/
TRAINER=FA3        # backbone
VERSION=VL         # V or VL
PROMPT=class       # class, engineering, ensembling
CFG=vit_amp        # config file
PROMTLENGTH='6'

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
    --preprocess ${PREPROCESS} \
    --seed ${SEED} \
    --LR ${LR} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --version ${VERSION} \
    --prompt ${PROMPT} \
    --prompt_length ${PROMTLENGTH} \
    # fi
done