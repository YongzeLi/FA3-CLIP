import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.cuda.amp import GradScaler, autocast
from functools import partial
from tqdm.notebook import tqdm
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from clip import clip
from trainers.FAS_trainer import TrainerFAS
from util.utils_FAS import cross_entropy
from trainers.funcTools import TextPart, HighFrequencyExtractor, nt_xent_loss, VisionPart


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

        print("Building custom CLIP@"+self.version)
        if 'VL' == self.version:
            self.model = clip_model
            if 'RN50' in cfg.MODEL.BACKBONE.NAME:
                embed_dim = 1024
            elif 'ViT-B' in cfg.MODEL.BACKBONE.NAME:
                embed_dim = 512
                vision_dim = 768
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
        self.model.highFrequencyExtractor = HighFrequencyExtractor()
        self.model.TextPart = TextPart(cfg, clip_model, self.classnames, embed_dim, vision_dim, self.device).to(torch.float32)
        self.nt_xent_loss = nt_xent_loss
        self.model.VisionPart = VisionPart(cfg, vision_dim, embed_dim, device=self.device).to(torch.float32)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            self.model.to(torch.float32)
        self.model.to(self.device)

        for name, param in self.model.named_parameters():
            param.requires_grad_(cfg.TRAINER.UPDATE)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        for param in self.model.transformer.parameters():
            param.requires_grad = False

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP@"+cfg.TRAINER.CLIP.VERSION, self.model, self.optim, self.sched)
        self.scaler = torch.GradScaler() if cfg.TRAINER.PREC == "amp" else None

    def forward_backward(self, batch):
        XY_R, XY_T, XY_L = self.parse_batch_train(batch)

        with torch.autocast(device_type='cuda'):
            if 'VL' == self.version:
                logit, loss_nt = self.forward_VL(self.model, XY_R, split='train')
            elif 'V' == self.version:
                image_features = self.model(XY_R)
                image_features = self.model.norm(image_features.float())
                logit = self.model.head(image_features)

        loss = F.cross_entropy(logit, XY_L) + loss_nt
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

    def model_inference(self, XY_R, split=None):
        with torch.autocast(device_type='cuda'):
            if 'VL' == self.version:
                logit = self.forward_VL(self.model, XY_R, split=split)
            elif 'V' == self.version:
                image_features = self.model(XY_R)
                image_features = self.model.norm(image_features.float())
                logit = self.model.head(image_features)
            return logit

    def forward_VL(self, clip_model, XY_R, split=None):
        logit_scale = self.logit_scale.exp()
        hq_XY_R = clip_model.highFrequencyExtractor(XY_R)
        img_cls, feq_each = clip_model.encode_image(XY_R) # Tensor: img feq features, List: each layer feq features
        image_features = clip_model.VisionPart(img_cls, feq_each, hq_XY_R)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_fake, text_features_live = clip_model.TextPart(XY_R, hq_XY_R, split)

        if split == 'train':
            fake_img = image_features[:int(image_features.shape[0]/2), :]
            live_img = image_features[int(image_features.shape[0]/2):, :]

            logits_FF = logit_scale * fake_img @ text_features_fake.t()
            logits_FL = logit_scale * fake_img @ text_features_live.t()

            logits_FF = torch.diag(logits_FF).unsqueeze(1)
            logits_FL = torch.diag(logits_FL).unsqueeze(1)
            logits_fake = torch.cat((logits_FF, logits_FL), dim=1)


            logits_LF = logit_scale * live_img @ text_features_fake.t()
            logits_LL = logit_scale * live_img @ text_features_live.t()

            logits_LF = torch.diag(logits_LF).unsqueeze(1)
            logits_LL = torch.diag(logits_LL).unsqueeze(1)
            logits_live = torch.cat((logits_LF, logits_LL), dim=1)

            logits = torch.cat((logits_fake, logits_live), dim=0)

            loss_NT_l = self.nt_xent_loss(text_features_live, text_features_live, text_features_fake, temperature=0.5)
            loss_NT_f = self.nt_xent_loss(text_features_fake, text_features_fake, text_features_live, temperature=0.5)
            loss_nt = loss_NT_l + loss_NT_f

            return logits, loss_nt

        elif split in ('test', 'val'):
            logits_fake_all = logit_scale * image_features @ text_features_fake.t()
            logits_live_all = logit_scale * image_features @ text_features_live.t()

            logits_fake = torch.diag(logits_fake_all).unsqueeze(1)
            logits_live = torch.diag(logits_live_all).unsqueeze(1)

            logits = torch.cat((logits_fake, logits_live), dim=1)

            return logits
        else:
            raise ValueError(f"please input correct mode (train, test or val) !!!")

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
