"""DINOv3 ViT-S/16 over log-mel inputs, preserving CA-TCC's [B,C,T] contract."""
from pathlib import Path
import importlib
import sys

import torch
from torch import nn
from torch.nn import functional as F


class DinoSequenceEncoder(nn.Module):
    def __init__(self, config, repo, weights=None, initialization='pretrained', image_height=64, clip_z=4.0):
        super().__init__()
        repo = Path(repo).expanduser().resolve()
        if not (repo / 'hubconf.py').is_file():
            raise FileNotFoundError(f'Provide a local official DINOv3 checkout containing hubconf.py: {repo}')
        if image_height < 16 or image_height % 16:
            raise ValueError('image_height must be a positive multiple of 16.')
        if clip_z <= 0 or config.features_len <= config.TC.timesteps:
            raise ValueError('clip_z must be positive; sequence length must exceed TC timesteps.')
        if initialization not in ('pretrained', 'random'):
            raise ValueError('initialization must be pretrained or random')
        if initialization == 'pretrained' and (not weights or not Path(weights).expanduser().is_file()):
            raise FileNotFoundError('Pretrained initialization requires a local official ViT-S/16 checkpoint.')
        if initialization == 'random' and weights:
            raise ValueError('Do not supply weights for the random-initialization control.')
        # No automatic clone or model download; the checkpoint is loaded explicitly below.
        # Import just the official backbone. hubconf.py also eagerly imports unrelated
        # detection/text heads and their optional dependencies (e.g. torchmetrics).
        sys.path.insert(0, str(repo))
        try:
            backbones = importlib.import_module('dinov3.hub.backbones')
            if not Path(backbones.__file__).resolve().is_relative_to(repo):
                raise RuntimeError('A different DINOv3 checkout is already imported in this process.')
            self.backbone = backbones.dinov3_vits16(pretrained=False)
        finally:
            sys.path.pop(0)
        if initialization == 'pretrained':
            state = torch.load(Path(weights).expanduser(), map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        if self.backbone.embed_dim != 384 or tuple(self.backbone.patch_embed.patch_size) != (16, 16):
            raise ValueError('This adapter supports official dinov3_vits16 (384 channels, 16x16 patches).')
        self.input_channels = config.input_channels
        self.image_height = image_height
        self.sequence_length = config.features_len
        self.image_width = 16 * self.sequence_length
        self.clip_z = float(clip_z)
        self.projection = nn.Conv1d(384, config.final_out_channels, kernel_size=1)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone_frozen = False

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError('Expected normalized log-mel tensor [batch, 40, time].')
        # Fixed scaling of TRAIN-normalized inputs; no per-image or test-set fitting.
        image = (x.clamp(-self.clip_z, self.clip_z) / self.clip_z + 1.0) * 0.5
        image = F.interpolate(image.unsqueeze(1), size=(self.image_height, self.image_width),
                              mode='bilinear', align_corners=False)
        image = (image.repeat(1, 3, 1, 1) - self.image_mean) / self.image_std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.backbone_frozen):
            tokens = self.backbone.forward_features(image)['x_norm_patchtokens']
        height, width = self.image_height // 16, self.sequence_length
        if tokens.shape[1] != height * width:
            raise RuntimeError('Unexpected DINOv3 patch grid; check upstream API/version.')
        # Patch order is row-major: frequency rows x time columns. Exclude CLS/registers.
        sequence = tokens.reshape(x.shape[0], height, width, 384).mean(dim=1)
        return self.projection(sequence.transpose(1, 2).float())


class DinoAudioModel(nn.Module):
    def __init__(self, config, repo, weights=None, initialization='pretrained', image_height=64,
                 clip_z=4.0, backbone_policy='trainable'):
        super().__init__()
        if backbone_policy not in ('trainable', 'frozen'):
            raise ValueError('backbone_policy must be trainable or frozen')
        self.encoder = DinoSequenceEncoder(config, repo, weights, initialization, image_height, clip_z)
        self.logits = nn.Linear(config.final_out_channels * config.features_len, config.num_classes)
        self.backbone_policy = backbone_policy
        self.linear_probe = False
        self.configure_stage('self_supervised')

    def configure_stage(self, training_mode):
        self.linear_probe = 'train_linear' in training_mode
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if self.linear_probe:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        freeze_backbone = self.linear_probe or self.backbone_policy == 'frozen'
        self.encoder.backbone_frozen = freeze_backbone
        if freeze_backbone:
            for parameter in self.encoder.backbone.parameters():
                parameter.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.linear_probe:
            self.encoder.eval()
        elif self.encoder.backbone_frozen:
            self.encoder.backbone.eval()
        return self

    def forward(self, x):
        features = self.encoder(x)
        return self.logits(features.flatten(1)), features
