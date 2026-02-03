import os.path as osp
import numpy as np
import copy
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from tqdm.notebook import tqdm
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from clip import clip
from trainers.FAS_trainer import TrainerFAS
from util.utils_FAS import cross_entropy
from collections import OrderedDict
from einops import rearrange
from timm.models.vision_transformer import Mlp


class SF_To_prompt(nn.Module):
    def __init__(self, cfg, embed_dim):
        super(SF_To_prompt, self).__init__()

        prompt_len = cfg.TRAINER.FA3.PROMPT_LENGTH
        self.embed_dim = embed_dim
       
        self.proj = nn.Linear(196, prompt_len)

        self.prompt_net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 512, 14, 14]
            nn.ReLU(inplace=True), 
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False),  # [B, 512, 14, 14]
            nn.ReLU(inplace=True), 
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 512, 14, 14]

            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 512, 14, 14]
            nn.ReLU(inplace=True), 
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False),  # [B, 512, 14, 14]
            nn.ReLU(inplace=True), 
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 512, 14, 14]
        )
        
    def forward(self, x): # [8, 512, 14, 14]
        B, D, H, W = x.shape  

        sp_fea = self.prompt_net(x) # [8, 512, 14, 14]
        out = self.proj(sp_fea.view(B, self.embed_dim, -1)) # [B, 512, 196] --> [B, 512, prompt_len]
        out = out.permute(0, 2, 1) # [B, prompt_len, 512]

        return out


class Prompter_learn(nn.Module):
    # init prompt
    # get prompted features
    def __init__(self, cfg, classnames, clip_model, device=None):
        super(Prompter_learn, self).__init__()
        n_cls = len(classnames)
        prompt_len = cfg.TRAINER.FA3.PROMPT_LENGTH
        dtype = clip_model.dtype
        prompt_dim = clip_model.ln_final.weight.shape[0] # 512
        prompt_prefix = " ".join(["X"] * prompt_len)

        self.ctx = nn.ParameterList([nn.Parameter(torch.empty(1, prompt_len, prompt_dim, dtype=dtype)) for _ in range(n_cls)]).to(device) 
        for single_para in self.ctx:
            print("single_para", single_para.shape)
            # print(single_para)
            nn.init.normal_(single_para, std=0.02)

        classnames = [name.replace("_", " ") for name in classnames] # [fake, live]
        prompts_fake = [prompt_prefix + " " + classnames[0] + "." ]
        prompts_live = [prompt_prefix + " " + classnames[1] + "."]

        tokenized_prompts_fake = torch.cat([clip.tokenize(p) for p in prompts_fake]) # [1，77]
        tokenized_prompts_live = torch.cat([clip.tokenize(p) for p in prompts_live]) # [1，77]
        with torch.no_grad():
            embedding_fake = clip_model.token_embedding(tokenized_prompts_fake.to(device)).type(dtype) # [1，77,512]
            embedding_live = clip_model.token_embedding(tokenized_prompts_live.to(device)).type(dtype) # [1，77,512]

        self.register_buffer("token_prefix_fake", embedding_fake[:, :1, :])  # SOS/start # These token vectors will be saved when in save_model(), but they should be ignored in load_model() as we want to use those computed using the current class names  NOTE: buffer - no grad      SOS, EOS, CLASS
        self.register_buffer("token_suffix_fake", embedding_fake[:, 1 + prompt_len:, :])  # CLS, EOS/end
        
        self.register_buffer("token_prefix_live", embedding_live[:, :1, :])
        self.register_buffer("token_suffix_live", embedding_live[:, 1 + prompt_len:, :])

        self.tokenized_prompts_fake = tokenized_prompts_fake
        self.tokenized_prompts_live = tokenized_prompts_live

    def forward(self, img_prompt_fake, img_prompt_live, split=None):
        B, prmot_len, C = img_prompt_fake.shape
        ori_fake = self.ctx[0].expand(B, -1, -1)  # [B, 12, 512]
        ori_live = self.ctx[1].expand(B, -1, -1)
         
        ctx_fake = img_prompt_fake + ori_fake # [B, 12, 512]  fake
        ctx_live = img_prompt_live + ori_live # [B, 12, 512]  live

        prefix_fake = self.token_prefix_fake.expand(B, -1, -1) # [1, 1, 512] -->  [B, 1, 512]
        suffix_fake = self.token_suffix_fake.expand(B, -1, -1) # [1, 64, 512] --> [B, 64, 512]
        prefix_live = self.token_prefix_live.expand(B, -1, -1)
        suffix_live = self.token_suffix_live.expand(B, -1, -1)

        prompts_fake = torch.cat(  # [B,77,512]
            [
                prefix_fake,  # [B, 1, 512]
                ctx_fake,     # [B, 12, 512]
                suffix_fake,  # [B, 60, 512]
            ], dim=1)

        prompts_live = torch.cat( 
            [
                prefix_live, 
                ctx_live,     
                suffix_live, 
            ], dim=1)
        
        tokenized_prompts_fake = self.tokenized_prompts_fake.expand(B, -1) # [B, 77]
        tokenized_prompts_live = self.tokenized_prompts_live.expand(B, -1) # [B, 77]
    
        return prompts_fake, prompts_live, tokenized_prompts_fake, tokenized_prompts_live # [B, 77, 512],[B, 77, 512]  , [B, 77], [B, 77]


class img_to_text_net(nn.Module):
    def __init__(self, SF_To_prompt_fake, SF_To_prompt_live, embed_dim):
        super(img_to_text_net, self).__init__()

        self.SF_To_prompt_fake = SF_To_prompt_fake
        self.SF_To_prompt_live = SF_To_prompt_live

        self.lamda1 = nn.Parameter(torch.tensor(-2.0))
        self.conv_Extr = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 56, 56]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256 28, 28]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), 
            
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1), # [B, 512, 14, 14]
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.Freconv_Extr = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 56, 56]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256 28, 28]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), 
            
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1), # [B, 512, 14, 14]
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_freq, split=None):
        
        x_img = self.conv_Extr(x.to(torch.float32))  # [B, 512, 14, 14]
        x_freq = self.Freconv_Extr(x_freq.to(torch.float32))  # [B, 512, 14, 14]
        
        lamda = torch.sigmoid(self.lamda1)
        x_input = lamda*x_img + (1-lamda)*x_freq

        if split != 'val' and split != 'test': 
            fake_fea = x_input[:int(x_input.shape[0]/2), :, :, :]  # [B/2, 3, 224, 224] 
            live_fea = x_input[int(x_input.shape[0]/2):, :, :, :]  # [B/2, 3, 224, 224] 

            img_prompt_fake = self.SF_To_prompt_fake(fake_fea.to(torch.float32))  # [B/2, 3, 224, 224] --> [B/2, prompt_length, 512]
            img_prompt_live = self.SF_To_prompt_live(live_fea.to(torch.float32))  # [B/2, 3, 224, 224] --> [B/2, prompt_length, 512]
        else:
            img_prompt_fake = self.SF_To_prompt_fake(x_input.to(torch.float32))  # [B/2, 3, 224, 224] --> [B/2, prompt_length, 512]
            img_prompt_live = self.SF_To_prompt_live(x_input.to(torch.float32))  # [B/2, 3, 224, 224] --> [B/2, prompt_length, 512]
        
        return img_prompt_fake, img_prompt_live


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts): # [B,77,512], [B,77]
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND [77, B, 512]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # B,77,512

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # [B,512] @ [512, 512] = [B, 512]

        return x


class TextPart(nn.Module):
    def __init__(self, cfg, clip_model, classnames, embed_dim, vision_dim, device):
        super().__init__()

        self.embed_dim = embed_dim
        self.vision_dim = vision_dim
        self.textEncoder = TextEncoder(clip_model)
        self.prompter_learn = Prompter_learn(cfg, classnames, clip_model, device=device)
        

        self.SF_To_prompt_fake = SF_To_prompt(cfg, embed_dim)
        self.SF_To_prompt_live = SF_To_prompt(cfg, embed_dim)
        self.img_to_text_net = img_to_text_net(self.SF_To_prompt_fake, self.SF_To_prompt_live, embed_dim)

    def forward(self, XY_R, hq_XY_R=None, split='train'): # [B,77,512], [B,77]
        if split != 'train' and split != 'test' and split != 'val':
            raise ValueError(f"please input correct (train or test or val) mode")

        total_prompt_fake, total_prompt_live = self.img_to_text_net(XY_R, hq_XY_R, split)
        prompts_fake, prompts_live, tokenized_prompts_fake, tokenized_prompts_live = self.prompter_learn(total_prompt_fake, total_prompt_live, split=split)
        text_features_fake = self.textEncoder(prompts_fake, tokenized_prompts_fake)
        text_features_live = self.textEncoder(prompts_live, tokenized_prompts_live)
        text_features_fake = text_features_fake / text_features_fake.norm(dim=-1, keepdim=True)
        text_features_live = text_features_live / text_features_live.norm(dim=-1, keepdim=True)
        
        return text_features_fake, text_features_live


def nt_xent_loss(anchor, positive, negatives, temperature=0.5):
    pos_sim = torch.matmul(anchor, positive.T)  # [batch_size, num_negatives]
    pos_sim = pos_sim / temperature
    pos_sim = pos_sim * (~torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device))

    neg_sim = torch.matmul(anchor, negatives.T)  # [batch_size, num_negatives]
    neg_sim = neg_sim / temperature

    pos_exp = torch.exp(pos_sim).sum(dim=-1)
    neg_exp = torch.exp(neg_sim).sum(dim=-1)

    loss = -torch.log(pos_exp / (pos_exp + neg_exp))  # [batch_size]

    return loss.mean()


class HighFrequencyExtractor(nn.Module):
    def __init__(self):
        super(HighFrequencyExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True), 
        )
        self.realconv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=False)
        self.imagconv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=False)


    def highFreqWH(self, x, scale):
        input_type = x.dtype
        x = x.to(torch.float32)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1])
        b,c,h,w = x.shape
        x[:, :, h//2-int(h//2*scale):h//2+int(h//2*scale), w//2-int(w//2*scale):w//2+int(w//2*scale)] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x.to(input_type)

    def forward(self, x):
        x = self.highFreqWH(x, 0.25)  # α
        x = self.conv1(x)
        x = x.to(torch.float32)
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]).to(torch.float32) 
        x = x.to(torch.complex64)

        real_part = self.realconv1(x.real).to(torch.float32)
        phase_part = self.imagconv1(x.imag).to(torch.float32)
        x = torch.complex(real_part, phase_part)
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        return x


class Attention(torch.nn.Module):
    def __init__(
            self, dim, num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def default(self, val, d):
        if val is not None:
            return val
        return d() if inspect.isfunction(d) else d
        
    def qkv_cal(self, q, k, v, mask=None):
        # [B, P, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = dots + mask
        attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]
        return out

    def forward(self, x, context=None, mask=None):
        # x [B, P_q, D]; context [B, P_kv, D]
        b, n, _ = x.shape
        kv_input = self.default(context, x)
        q_input = x

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        out = self.qkv_cal(q, k, v, mask)
        # [B, P, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class STMAttention(Attention):
    def forward(self, q_input, kv_input, token_score):
        # q [B, P_q, D]; kv [B, P_kv, D]; token_score [B, P_kv, 1]
        b, n, _ = q_input.shape
        
        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        # qkv_cal
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        token_score = token_score.squeeze(-1)[:, None, None, :]  # [B, 1, 1, P_kv]
        attn = (dots + token_score).softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class STMBlock(torch.nn.Module):
    def __init__(self,
                 dim,
                 out_token_len,
                 k,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 act_layer=torch.nn.GELU,
                 norm_layer=torch.nn.LayerNorm):
        super(STMBlock, self).__init__()
        self.dim = dim
        self.norm1 = torch.nn.LayerNorm(dim)

        # cluster
        self.out_token_len = out_token_len
        self.k = k

        # merger
        self.score_mlp = torch.nn.Linear(dim, 1)

        # transformer block
        self.attention_layer = \
            STMAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        
    def _index_points(self, points, idx):
        """Sample features following the index.
            Returns:
                new_points:, indexed points data, [B, S, C]

            Args:
                points: input points data, [B, N, C]
                idx: sample index data, [B, S]
            """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def cluster(self, feature, batch_size):
        with torch.no_grad():
            distance_matrix = torch.cdist(feature, feature) / (self.dim ** 0.5)

            # get local density
            distance_nearest, index_nearest = \
                torch.topk(distance_matrix, k=self.k, dim=-1, largest=False)
            density = (-(distance_nearest ** 2).mean(dim=-1)).exp()

            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(feature.dtype)
            dist_max = distance_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (distance_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=self.out_token_len, dim=-1)

            # assign tokens to the nearest center
            distance_matrix = self._index_points(distance_matrix, index_down)

            idx_cluster = distance_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = \
                torch.arange(batch_size, device=feature.device)[:, None].expand(batch_size, self.out_token_len)
            idx_tmp = \
                torch.arange(self.out_token_len, device=feature.device)[None, :].expand(batch_size, self.out_token_len)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster

    def merger(self, feature, idx_cluster, token_score, batch_size, patch_num):
        idx_batch = torch.arange(batch_size, device=feature.device)[:, None]
        idx = idx_cluster + idx_batch * self.out_token_len

        token_weight = token_score.exp()
        all_weight = token_weight.new_zeros(batch_size * self.out_token_len, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(batch_size * patch_num),
                              source=token_weight.reshape(batch_size * patch_num, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # average token features
        merged_feature = feature.new_zeros(batch_size * self.out_token_len, self.dim)
        source = feature * norm_weight
        merged_feature.index_add_(dim=0, index=idx.reshape(batch_size * patch_num),
                                  source=source.reshape(batch_size * patch_num, self.dim).type(feature.dtype))
        merged_feature = merged_feature.reshape(batch_size, self.out_token_len, self.dim)

        return merged_feature

    def transformer_block(self, q_input, kv_input, token_score): 
        attn = self.attention_layer(q_input, kv_input, token_score) # STMAttention
        feature = q_input + attn
        feature = feature + self.mlp(self.norm2(feature))
        return feature
        
    def forward(self, x): # [B, 196, 512]
        # feature [B, V*P, D]
        batch_size, patch_num, _ = x.shape

        x = self.norm1(x)
        token_score = self.score_mlp(x) # [B, 196, 1]
    
        idx_cluster = self.cluster(x, batch_size) # [B, 196]
        q_input = self.merger(x, idx_cluster, token_score, batch_size, patch_num) # [B, OUT_TOKEN_LEN, D]
        kv_input = x
    
        feature = self.transformer_block(q_input, kv_input, token_score)  # [B, OUT_TOKEN_LEN, D]
    
        return feature


class VisionPart(nn.Module):
    def __init__(self, cfg, clip_model, vision_dim, embed_dim, device='cuda'):
        super().__init__()

        self.encode_image = clip_model.visual
        self.dtype = clip_model.dtype
        self.vision_dim = vision_dim
        scale = vision_dim ** -0.5
        self.STM_LEN = cfg.TRAINER.FA3.STM_LEN
        self.K_NUM = cfg.TRAINER.FA3.K_NUM
        self.NUM_HEAD = cfg.TRAINER.FA3.NUM_HEAD
        
        self.lamdaFreq = nn.Parameter(torch.tensor(-1.89))
        self.proj = nn.Parameter(scale * torch.randn(self.vision_dim, embed_dim))

        self.conv_ori = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, self.vision_dim, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(self.vision_dim),
            nn.ReLU(inplace=True)
        )
        # Extract frequency features     
        self.realconv = nn.Sequential(*[nn.Sequential(nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, stride=1, bias=False)) for _ in range(12)])
        self.imagconv = nn.Sequential(*[nn.Sequential(nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, stride=1, bias=False)) for _ in range(12)])
        self.beforeFFT = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            ) for _ in range(12)])
        self.afterFFT = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=3, stride=1, padding=1, bias=False),
            ) for _ in range(12)])

        self.StmBlock = STMBlock(dim=self.vision_dim, out_token_len=self.STM_LEN, k=self.K_NUM, num_heads=self.NUM_HEAD)
        self.score1_mlp = torch.nn.Linear(self.vision_dim, 1)
    
    def highFreqWH(self, x, scale): # [B, 768, 14, 14]
        input_type = x.dtype
        x = x.to(torch.float32)
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        b,c,h,w = x.shape
        x[:, :, h//2-int(h//2*scale):h//2+int(h//2*scale), w//2-int(w//2*scale):w//2+int(w//2*scale)] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])  
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x.to(input_type)
    
    # x: 12*[196, B, 768], x_ori: [B, 3, 224, 224]
    def forward(self, x_ori: torch.Tensor):
        img_cls, x = self.encode_image(x_ori.type(self.dtype))
        freq_list = []
        for i in range(12): 
            x_i = copy.deepcopy(x[i].detach())
            x_i = x_i.permute(1, 2, 0) # [B, 768, 196]
            x_i = x_i.reshape(-1, x_i.shape[1], int(x_i.shape[2]**0.5), int(x_i.shape[2]**0.5))

            x_high = self.highFreqWH(x_i, 0.25).to(torch.float32)
            x_bef = self.beforeFFT[i](x_high)
            x_bef = x_bef + x_i
            x_bef = x_bef.to(torch.float32)

            x_i_freq = torch.fft.fft2(x_bef, norm="ortho")
            x_i_freq = torch.fft.fftshift(x_i_freq, dim=[2, 3]).to(torch.float32)  # [B, 768, 14, 14]
            x_i_freq = x_i_freq.to(torch.complex64)
 
            x_i_freq_real = self.realconv[i](x_i_freq.real).to(torch.float32)  # [B, 768, 14, 14]
            x_i_freq_imag = self.imagconv[i](x_i_freq.imag).to(torch.float32)
            freq_i = torch.complex(x_i_freq_real, x_i_freq_imag)

            freq_i = torch.fft.ifftshift(freq_i, dim=[2, 3])
            freq_i = torch.fft.ifft2(freq_i, norm="ortho")
            freq_i = torch.real(freq_i) # [B, 768, 14, 14]
            freq_i = F.relu(freq_i, inplace=True)
            x_space = self.afterFFT[i](freq_i).to(torch.float32)
            x_space = x_space + x_bef

            x_space = x_space.reshape(x_space.shape[0], self.vision_dim, -1) # [B, 768, 196]
            x_space = x_space.permute(0, 2, 1).to(torch.float32)  # [B, 196, 768]

            freq_list.append(x_space) # [B, 196, 768]

        x_ori = self.conv_ori(x_ori) # [B, 3, 224, 224]-->[B, 768, 14, 14]
        x_ori = x_ori.reshape(x_ori.shape[0], x_ori.shape[1], -1) # [B, 768, 196]
        x_ori = x_ori.permute(0, 2, 1) # [B, 196, 768]

        x_freq_concat = torch.cat(freq_list, dim=1)  
        all_freq = torch.cat([x_ori, x_freq_concat], dim=1)     # [B, 196*13, 768]
        fea_stm = self.StmBlock(all_freq) # [B, 196*13, 768]-->[B, 32, 768]        

        stm_wight = self.score1_mlp(fea_stm) # [B, 32, 768] --> [B, 32, 1]
        fea_sum = torch.sum(fea_stm * stm_wight, dim=1)  # [B, 768]
        fea_sum = fea_sum @ self.proj # [B, 768] @ [768, 512] --> [B, 512]

        lamdaFreq = torch.sigmoid(self.lamdaFreq)
        out = img_cls + lamdaFreq*fea_sum # [B, 512]

        return out


class CustomCLIP(nn.Module):
    def __init__(self, cfg, device, classnames, clip_model, embed_dim, vision_dim):
        super().__init__()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip_model = clip_model
        self.highFrequencyExtractor = HighFrequencyExtractor()
        self.VisionPart = VisionPart(cfg, clip_model, vision_dim, embed_dim, device)
        self.TextPart = TextPart(cfg, clip_model, classnames, embed_dim, vision_dim, device)
        self.nt_xent_loss = nt_xent_loss

    def forward(self, XY_R, split=None):
        logit_scale = self.logit_scale.exp()
        # image branch
        image_features = self.VisionPart(XY_R) # [B, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [B, 512]
         # text branch
        hq_XY_R = self.highFrequencyExtractor(XY_R)  # [B, 3, 224, 224] --> [B, 3, 224, 224]
        text_features_fake, text_features_live = self.TextPart(XY_R, hq_XY_R, split) # train: [B/2, 512], test: [B, 512]

        if split == 'train':  
            fake_img = image_features[:int(image_features.shape[0]/2), :]  # [B/2, 512]
            live_img = image_features[int(image_features.shape[0]/2):, :]  # [B/2, 512]

            logits_FF = logit_scale * fake_img @ text_features_fake.t() # [B/2, 512] @ [512, B/2] = [B/2, B/2]
            logits_FL = logit_scale * fake_img @ text_features_live.t() # [B/2, 512] @ [512, B/2] = [B/2, B/2]

            logits_FF = torch.diag(logits_FF).unsqueeze(1) # [B/2, 1]
            logits_FL = torch.diag(logits_FL).unsqueeze(1) # [B/2, 1]
            logits_fake = torch.cat((logits_FF, logits_FL), dim=1)  # [B/2, 2]

            logits_LF = logit_scale * live_img @ text_features_fake.t() # [B/2, 512] @ [512, B/2] = [B/2, B/2]
            logits_LL = logit_scale * live_img @ text_features_live.t() # [B/2, 512] @ [512, B/2] = [B/2, B/2]

            logits_LF = torch.diag(logits_LF).unsqueeze(1) # [B/2, 1]
            logits_LL = torch.diag(logits_LL).unsqueeze(1) # [B/2, 1]
            logits_live = torch.cat((logits_LF, logits_LL), dim=1)  # [B/2, 2]

            logits = torch.cat((logits_fake, logits_live), dim=0)  # [B, 2]

            # NT-Xent
            loss_NT_l = self.nt_xent_loss(text_features_live, text_features_live, text_features_fake, temperature=0.5)
            loss_NT_f = self.nt_xent_loss(text_features_fake, text_features_fake, text_features_live, temperature=0.5)
            loss_nt = loss_NT_l + loss_NT_f

            return logits, loss_nt

        elif split == 'test' or split == 'val':
            logits_fake_all = logit_scale * image_features @ text_features_fake.t() # [B, 512] @ [512, B] = [B, B]
            logits_live_all = logit_scale * image_features @ text_features_live.t() # [B, 512] @ [512, B] = [B, B]

            logits_fake = torch.diag(logits_fake_all).unsqueeze(1) # [B, 1]
            logits_live = torch.diag(logits_live_all).unsqueeze(1) # [B, 1]

            logits = torch.cat((logits_fake, logits_live), dim=1)  # [B, 2]

            return logits
        else:
            raise ValueError(f"please input correct mode (train, test or val) !!!")


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'FA3'
                      }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


@TRAINER_REGISTRY.register()
class FA3(TrainerFAS):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FA3.PREC in ["fp16", "fp32", "amp"]

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
        self.embed_dim = 512
        self.vision_dim = 768

    def build_model(self):
        cfg = self.cfg
        self.prec = cfg.TRAINER.FA3.PREC
        self.device = torch.device('cuda:%d' % cfg.TRAINER.GPU[0])
        self.version = cfg.TRAINER.FA3.VERSION

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model = clip_model.to(self.device)
        self.dtype = clip_model.dtype
        # CLIP@VL: Use its image encoder V and the text encoder L.
        print("Building custom CLIP@"+self.version)  ## for k, v in self.model.state_dict().items():
        
        self.model = CustomCLIP(cfg, self.device, self.classnames, clip_model, self.embed_dim, self.vision_dim)
        self.logit_scale = clip_model.logit_scale
        self.model.head = nn.Linear(self.embed_dim, 2, bias=True)
        self.model.norm = partial(nn.LayerNorm, eps=1e-6)(self.embed_dim)

        if self.prec == "fp32" or self.prec == "amp":
            self.model.to(torch.float32)
        self.model.to(self.device)

        for name, param in self.model.named_parameters():
            param.requires_grad_(cfg.TRAINER.UPDATE)

        # freeze text encoder
        for param in self.model.TextPart.textEncoder.transformer.parameters():
            param.requires_grad = False

        # Double check
        print("Trainable parameters:")
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                print(name)

        # NOTE: only give xxx to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP@"+cfg.TRAINER.CLIP.VERSION, self.model, self.optim, self.sched)
        self.scaler = torch.GradScaler() if cfg.TRAINER.FA3.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
    
    def forward_backward(self, batch):
        XY_R, XY_T, XY_L = self.parse_batch_train(batch)

        with torch.autocast(device_type='cuda'):
            logit, loss_nt = self.model(XY_R, split='train')

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
            logit = self.model(XY_R, split=split)

        return logit

    def parse_batch_train(self, batch):
        X_R, X_T, X_L = batch['X_R'].to(self.device), batch['X_T'], batch['X_L'].to(self.device) # X_R fake sample，label 0
        Y_R, Y_T, Y_L = batch['Y_R'].to(self.device), batch['Y_T'], batch['Y_L'].to(self.device) # Y_R live sample，label 1

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
