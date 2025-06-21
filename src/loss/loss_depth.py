from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..geometry.projection import transform_world2cam
from ..Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

@dataclass
class LossDepthCfg:
    weight: float                             # overall scaling for depth loss
    trim_ratio: float = 0.05                  # fraction of worst residuals to drop
    grad_weight: float = 1.5                  # weight for gradient loss term
    hinge_weight: float = 4.0                # weight for area-based barrier term
    allowed_negative_frac: float = 0.001       # tolerated fraction of negative pixels
    num_scales: int = 4                       # number of scales for gradient loss

@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg

class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load DepthAnythingV2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = 'vitb'
        self.model = DepthAnythingV2(**model_configs[self.encoder]).to(self.DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.load_state_dict(
            torch.load(f'checkpoints/depth_anything_v2_{self.encoder}.pth', map_location=self.DEVICE)
        )

    def _normalize_ssi(self, depth: Tensor) -> Tensor:
        flat = depth.flatten()
        med = flat.median()
        mad = (flat - med).abs().mean().clamp(min=1e-3)
        return (depth - med) / (mad + 1e-6)

    def _trimmed_mae(self, pred: Tensor, gt: Tensor) -> Tensor:
        res = (pred - gt).abs().flatten()
        keep = int(res.numel() * (1 - self.cfg.trim_ratio))
        trimmed, _ = torch.topk(res, keep, largest=False)
        return trimmed.mean()

    def _gradient_loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        loss = 0.0
        for s in range(self.cfg.num_scales):
            factor = 2 ** s
            p_s = F.interpolate(pred.unsqueeze(0).unsqueeze(0), scale_factor=1/factor, mode='bilinear', align_corners=False).squeeze()
            g_s = F.interpolate(gt.unsqueeze(0).unsqueeze(0), scale_factor=1/factor, mode='bilinear', align_corners=False).squeeze()
            d = p_s - g_s
            dx = (d[:, 1:] - d[:, :-1]).abs().mean()
            dy = (d[1:, :] - d[:-1, :]).abs().mean()
            loss += dx + dy
        return loss / self.cfg.num_scales

    def _area_barrier(self, pred: Tensor, valid: Tensor) -> Tensor:
        """
        Penalize only if a noticeable fraction of valid pixels are negative.
        """
        eps = self.cfg.allowed_negative_frac
        neg = (pred < 0) & valid
        frac_neg = neg.float().sum() / valid.float().sum()
        return F.relu(frac_neg - eps)

    def _neg_magnitude_penalty(self, pred: Tensor, valid: Tensor) -> Tensor:
        """
        Penalize the average magnitude of negative predictions over valid pixels.
        """
        mag = torch.relu(-pred)
        return mag[valid].mean()

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int
    ) -> Float[Tensor, ""]:
        B, V, H, W = prediction.depth.shape[0], 3, 256, 256
        accum = {'ssi': 0.0, 'grad': 0.0, 'hinge': 0.0}
        count = 0

        for b in range(B):
            for v in range(V):
                d_pred = prediction.depth[b,v,:252,:252]

                img = batch['target']['image'][b, v, :, :252, :252].unsqueeze(0)
                with torch.no_grad():
                    gt_np = self.model.infer_image(img, 518)
                raw = torch.from_numpy(gt_np).to(d_pred.device).clamp(min=1e-3)
                d_gt = 1.0 / raw

                valid = torch.isfinite(d_pred) & torch.isfinite(d_gt)
                if not valid.any():
                    print(f"[DepthLoss] No valid pixels for batch {b}, view {v}, skipping this view.")
                    continue

                # SSI term
                p_vals = d_pred[valid]
                g_vals = d_gt[valid]
                med_p, med_g = p_vals.median(), g_vals.median()
                mad_p = (p_vals - med_p).abs().mean().clamp(min=1e-6)
                mad_g = (g_vals - med_g).abs().mean().clamp(min=1e-6)
                p_norm = (d_pred - med_p) / mad_p
                g_norm = (d_gt - med_g) / mad_g
                accum['ssi'] += self._trimmed_mae(p_norm[valid], g_norm[valid])

                # gradient term
                accum['grad'] += self._gradient_loss(p_norm, g_norm)

                # combine area and magnitude penalties
                area_term = self._area_barrier(d_pred, valid)
                accum['hinge'] += area_term
                count += 1

        if count == 0:
            print("[DepthLoss] No valid views in this batch. Returning zero loss.")
            return torch.tensor(0.0, device=self.DEVICE)

        accum = {k: accum[k] / count for k in accum}
        loss = (
            accum['ssi']
            + self.cfg.grad_weight * accum['grad']
            + self.cfg.hinge_weight * accum['hinge']
        )
        return loss * self.cfg.weight / 40.0
