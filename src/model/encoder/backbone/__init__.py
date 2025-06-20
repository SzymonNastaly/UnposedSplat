from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_fast3r import BackboneFast3R, BackboneFast3RCfg
from .backbone_vggt import BackboneVggt, BackboneVggtCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    "fast3r": BackboneFast3R,
    "vggt": BackboneVggt 
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg | BackboneFast3RCfg | BackboneVggtCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
