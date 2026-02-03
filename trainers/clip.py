import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from functools import partial
from tqdm.notebook import tqdm
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from clip import clip

from trainers.FAS_trainer import TrainerFAS
from util.utils_FAS import cross_entropy

@TRAINER_REGISTRY.register()
class CLIP(TrainerFAS):
    """CLIP@V: Use only its image encoder V and discard the text encoder L.
       CLIP@VL: Use its image encoder V and the text encoder L.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]  ## fp16

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """

        dataset = build_dataset(self.cfg)
        self.train_loader_x = dataset.train_loader_x
        self.val_loader = dataset.val_loader
        self.test_loader = dataset.test_loader
        self.dataset = dataset
        self.lab2cname = dataset.lab2cname
        self.classnames = dataset.classnames
        self.templates = dataset.templates

    def build_model(self):
        cfg = self.cfg
        self.device = torch.device('cuda:%d' % cfg.TRAINER.GPU[0])
        self.version = cfg.TRAINER.CLIP.VERSION
        self.prompt = cfg.TRAINER.CLIP.PROMPT

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)
        self.dtype = clip_model.dtype

        print("Building custom CLIP@"+self.version)  ## for k, v in self.model.state_dict().items():
        if 'VL' == self.version:
            self.model = clip_model
            if 'RN50' in cfg.MODEL.BACKBONE.NAME:
                # self.model.visual.attnpool = None
                embed_dim = 1024
            elif 'ViT-B' in cfg.MODEL.BACKBONE.NAME:
                # self.model.visual.proj = None
                embed_dim = 512
            self.logit_scale = clip_model.logit_scale
        elif 'V' == self.version:
            self.model = clip_model.visual
            if 'RN50' in cfg.MODEL.BACKBONE.NAME:
                self.model.attnpool = None
                embed_dim = 1024
            elif 'ViT-B' in cfg.MODEL.BACKBONE.NAME:
                self.model.proj = None
                embed_dim = 768
        self.model.head = nn.Linear(embed_dim, 2, bias=True)
        self.model.norm = partial(nn.LayerNorm, eps=1e-6)(embed_dim)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            self.model.float()

        for name, param in self.model.named_parameters():
            param.requires_grad_(cfg.TRAINER.UPDATE)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)
        # NOTE: only give xxx to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP@"+cfg.TRAINER.CLIP.VERSION, self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        XY_R, XY_T, XY_L = self.parse_batch_train(batch)
        with autocast():
            if 'VL' == self.version:
                # labels = torch.tensor(np.arange(XY_R.shape[0]), device=self.device)
                # logits_per_image, logits_per_text = self.model(XY_R, XY_T)
                # loss_i = F.cross_entropy(logits_per_image, labels)
                # loss_t = F.cross_entropy(logits_per_image, labels)
                # loss = (loss_i + loss_t) / 2.0
                logit = self.forward_VL(self.model, XY_R, text_features=None)
            elif 'V' == self.version:
                # if self.cfg.TRAINER.PREC == "amp":
                image_features = self.model(XY_R)
                image_features = self.model.norm(image_features.float())
                logit = self.model.head(image_features)
        

        loss = F.cross_entropy(logit, XY_L)

        self.optim.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "acc": cross_entropy(logit, XY_L)[0]
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, XY_R):
        with autocast():
            if 'VL' == self.version:
                logit = self.forward_VL(self.model, XY_R, text_features=None)
            elif 'V' == self.version:
                image_features = self.model(XY_R)
                image_features = self.model.norm(image_features.float())
                logit = self.model.head(image_features)
            return logit

    def forward_VL(self, clip_model, XY_R, text_features=None):
        logit_scale = self.logit_scale.exp()
        image_features = clip_model.encode_image(XY_R)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            text_descriptions = [f"This is an image of a {label} face" for label in self.classnames]
            prompts = clip.tokenize(text_descriptions).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit = logit_scale * image_features @ text_features.t()
        return logit

    def parse_batch_train(self, batch):
        X_R, X_T, X_L = batch['X_R'].to(self.device), batch['X_T'], batch['X_L'].to(self.device)
        Y_R, Y_T, Y_L = batch['Y_R'].to(self.device), batch['Y_T'], batch['Y_L'].to(self.device)

        XY_R = torch.cat([X_R, Y_R], dim=0)
        XY_T = X_T + Y_T
        XY_L = torch.cat([X_L, Y_L], dim=0)

        XY_R = XY_R.type(self.dtype)
        XY_T = clip.tokenize(XY_T).to(self.device)
        return XY_R, XY_T, XY_L

    def parse_batch_test(self, batch):
        frame, label, text = batch['frame'].to(self.device), batch['label'].to(self.device), batch['text']
        frame = frame.type(self.dtype)
        text = clip.tokenize(text).to(self.device)
        return frame, label