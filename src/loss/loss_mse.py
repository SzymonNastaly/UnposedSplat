from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    l1_loss: bool = False
    clamp_large_error: float = 0.1
    clamp_after_step: int = 1000


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        if self.cfg.clamp_large_error > 0 and global_step >= self.cfg.clamp_after_step:
            valid_mask = (delta ** 2) < self.cfg.clamp_large_error
            delta = delta[valid_mask]
            # If no pixels are valid, return a very small loss
            if delta.numel() == 0:
                return torch.tensor(1e-6, dtype=torch.float32, device=delta.device)
        if self.cfg.l1_loss:
            return self.cfg.weight * (delta.abs()).mean()
        return self.cfg.weight * (delta**2).mean()
