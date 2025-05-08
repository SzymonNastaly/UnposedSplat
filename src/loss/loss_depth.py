from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor
import numpy as np

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

from ..Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = 'vitb'

        self.model = DepthAnythingV2(**self.model_configs[self.encoder])
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
        self.model = self.model.to(self.DEVICE).eval()

    def forward(self, prediction: DecoderOutput, batch: BatchedExample, gaussians: Gaussians, global_step: int) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes
        near = batch["target"]["near"][..., None, None].log()
        far = batch["target"]["far"][..., None, None].log()
        
        # Get predicted depth and normalize in log space
        pred_depth = prediction.depth.minimum(far).maximum(near)
        pred_depth = (pred_depth - near) / (far - near)
        
        # Process the first image in the batch
        b_idx = 0
        first_image = batch['target']['image'][b_idx, 0, :, :252, :252]
        first_image = first_image.unsqueeze(0)  # Add batch dimension
        
        # Get GT depth from model
        gt_depth_np = self.model.infer_image(first_image, 518)
        
        # Convert NumPy array to PyTorch tensor and move to same device as pred_depth
        gt_depth = torch.from_numpy(gt_depth_np).to(pred_depth.device)
        
        # Normalize GT depth to [0,1]
        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        
        # Crop predicted depth
        pred_depth_cropped = pred_depth[b_idx, :, :252, :252]
        
        # Take the first channel of predicted depth
        pred_depth_single = pred_depth_cropped[0]  # Shape becomes [252, 252]
        
        # Ensure both have same shape
        assert gt_depth.shape == pred_depth_single.shape, \
            f"Shape mismatch: gt={gt_depth.shape}, pred={pred_depth_single.shape}"
        
        # Compute L1 loss
        return self.cfg.weight * (gt_depth - pred_depth_single).abs().mean()
