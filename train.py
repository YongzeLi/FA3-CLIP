import argparse
import torch
from datetime import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.UniAttackData
import util.evaluator
import trainers.clip
import trainers.fa3


current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.protocol:
        cfg.DATASET.PROTOCOL = args.protocol

    if args.protocol:
        cfg.DATASET.PREPROCESS = args.preprocess

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir + formatted_time

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.TEST.FINAL_MODEL = 'best_val'
    cfg.TEST.EVALUATOR = "FAS_Classification"

    if args.LR:
        cfg.OPTIM.LR = args.LR
    

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER = CN()
    cfg.TRAINER.GPU = [int(s) for s in args.gpu_ids.split(',')]

    cfg.TRAINER.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.UPDATE = True

    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.VERSION = args.version
    cfg.TRAINER.CLIP.PROMPT = args.prompt

    ## Baseline
    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.VERSION = args.version
    cfg.TRAINER.CLIP.PROMPT = args.prompt
    cfg.TRAINER.CLIP.PREC = "amp"
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16        # number of context vectors
    cfg.TRAINER.COOP.CTX_INIT = ""     # initialization words
    cfg.TRAINER.COOP.CSC = False       # class-specific context (False or True)
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = args.ctp
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Setup for FA3-CLIP
    cfg.TRAINER.FA3 = CN()
    cfg.TRAINER.FA3.VERSION = args.version
    cfg.TRAINER.FA3.STM_LEN = args.STM_LEN
    cfg.TRAINER.FA3.K_NUM = args.K_NUM
    cfg.TRAINER.FA3.NUM_HEAD = args.NUM_HEAD
    cfg.TRAINER.FA3.PROMPT_LENGTH = args.prompt_length
    cfg.TRAINER.FA3.PREC = "amp"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        _, threshold_val = trainer.test(split="val", thr=None)
        trainer.test(split="test", thr=threshold_val)
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu")
    parser.add_argument("--root", type=str, default="/sdh/UniAttackData", help="path to dataset")
    parser.add_argument("--protocol", type=str, default="p1@train@val@test", help="protocol")
    parser.add_argument("--preprocess", type=str, default="resize_crop_rotate_flip", help="preprocess")
    parser.add_argument("--output-dir", type=str, default="./test/", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="./configs/trainers/FA3-CLIP/vit_amp.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="./configs/datasets/UniAttackData.yaml", help="path to config file for dataset setup")
    parser.add_argument("--trainer", type=str, default="FA3", help="name of trainer")
    parser.add_argument("--version", type=str, default="VL", help="version of trainer")
    parser.add_argument("--prompt", type=str, default="class", help="type of text")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="./test/best_model", help="load model from this directory for eval-only mode",)
    parser.add_argument("--ctp", type=str, default="", help="class token position (end or middle)")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--LR", type=float, default=0.000001, help="learning rate")
    parser.add_argument("--this_code_set", type=str, default="", help="describe")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    # FA3-CLIP
    parser.add_argument("--prompt_length", type=int, default=6, help="learnable prompt length")
    parser.add_argument("--STM_LEN", default=32, help="center number of stm module")
    parser.add_argument("--K_NUM", type=int, default=15, help="k number of stm module")
    parser.add_argument("--NUM_HEAD", type=int, default=12, help="head number of stm module")

    args = parser.parse_args()
    main(args)
