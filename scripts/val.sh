#!/bin/bash
# custom config
# val checkpoint path:
# test/
# └── p1/
#     └── CLIP@VL/
#         └── model-best.pth.tar

DATA=/mnt/sdh/UniAttackData  # Dataset dir
MODELDIR=./test/p1  # load pth dir
OUTPUT=./test/  # pth save pth
PROTOCOL=p1@train@val@test # {protocol name}@train@val@test
GPU_IDS='0' # str
SEED=1
TRAINER=CLIP  # backbone
VERSION=VL # V or VL
PROMPT=class # class, engineering, ensembling
CFG=vit_b16 # config file
PROMPTLENGTH=6 # prompt length
LEARNINGRATE=0.000001

python ./train.py \
--gpu_ids ${GPU_IDS} \
--root ${DATA} \
--protocol ${PROTOCOL} \
--seed ${SEED} \
--trainer ${TRAINER} \
--config-file ./configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${OUTPUT} \
--version ${VERSION} \
--prompt ${PROMPT} \
--prompt_length ${PROMPTLENGTH} \
--LR ${LEARNINGRATE} \
--model-dir ${MODELDIR} \
--no-train \
--eval-only \
