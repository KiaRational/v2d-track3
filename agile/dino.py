"""DINOv3 dense features for AGILE's semantic loss (Eq. 4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Dino:
    def __init__(self, model_dir: str, device="cuda", input_res=448):
        # fp32: DINOv3 activations overflow fp16 (NaN features).
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
        self.patch = self.model.config.patch_size
        self.n_skip = 1 + getattr(self.model.config, "num_register_tokens", 0)  # CLS + registers
        self.res = input_res
        self.device = device

    @torch.no_grad()
    def features(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (H,W,3) in [0,1] -> L2-normalised patch features (Hf, Wf, C)."""
        x = rgb.permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        x = (x - MEAN.to(x)) / STD.to(x)
        tokens = self.model(pixel_values=x).last_hidden_state[0, self.n_skip:]
        g = self.res // self.patch
        return F.normalize(tokens.float().view(g, g, -1), dim=-1)


def _grid(uv: torch.Tensor, image_res: int) -> torch.Tensor:
    return uv / image_res * 2 - 1  # crop pixels -> [-1, 1] (align_corners=False)


def sample_features(fmap: torch.Tensor, uv: torch.Tensor, image_res: int) -> torch.Tensor:
    """Bilinearly sample a (Hf,Wf,C) feature map at crop pixels uv (N,2) -> (N,C)."""
    out = F.grid_sample(fmap.permute(2, 0, 1)[None], _grid(uv, image_res).view(1, -1, 1, 2), align_corners=False)
    return out[0, :, :, 0].T


def sample_maps(maps: torch.Tensor, uv: torch.Tensor, image_res: int) -> torch.Tensor:
    """Sample similarity map i (N,Hf,Wf) at its own point uv_i (N,2) -> (N,).
    Differentiable w.r.t. uv: this is how gradients reach the object pose in Eq. 4."""
    n = maps.shape[0]
    out = F.grid_sample(maps[:, None], _grid(uv, image_res).view(n, 1, 1, 2), align_corners=False,
                        padding_mode="zeros")
    return out.view(n)
