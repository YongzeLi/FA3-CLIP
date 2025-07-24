#!/bin/bash
# custom config

GPU_IDS='1'
LR='0.000001'
DATASET=UniAttackData
DATA=/mnt/sdh/UniAttackData
PROTOCOL=p1@train@val@test  # e.g. {protocol name}@train@val@test
PREPROCESS=resize_crop_rotate_flip   ### resize_crop_rotate_flip_ColorJitter
OUTPUT=./test
TRAINER=CLIP
VERSION=VL         # V or VL
PROMPT=class  # class, engineering, ensembling
CFG=vit_b16        # config file
PROMPTLENGTH='6'  # prompt length

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}@${PROMPT}/${CFG}/${PROTOCOL}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python ../train.py \
    --gpu_ids ${GPU_IDS} \
    --root ${DATA} \
    --protocol ${PROTOCOL} \
    --preprocess ${PREPROCESS} \
    --seed ${SEED} \
    --LR ${LR} \
    --trainer ${TRAINER} \
    --dataset-config-file ../configs/datasets/${DATASET}.yaml \
    --config-file ../configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --version ${VERSION} \
    --prompt ${PROMPT} \
    --prompt_length ${PROMPTLENGTH} \
    # fi
done