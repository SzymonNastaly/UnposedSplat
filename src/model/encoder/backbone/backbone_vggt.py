from .vggt.vggt.models.aggregator import Aggregator
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn.functional as F

inf = float('inf')

#FIXME: these params should be read from some config file
fast3r_params = {
    'encoder_args' : {
        "attn_implementation": "flash_attention",
        "depth": 24,
        "embed_dim": 1024,
        "encoder_type": "croco",
        "img_size": 512,
        "mlp_ratio": 4,
        "num_heads": 16,
        "patch_embed_cls": "PatchEmbedDust3R",
        "patch_size": 16,
        "pos_embed": "RoPE100"
    },
    'decoder_args' : {
        "attn_bias_for_inference_enabled": False,
        "attn_drop": 0.0,
        "attn_implementation": "flash_attention",
        "decoder_type": "fast3r_plus", # fast3r_plus or fast3r
        "depth": 24,
        "drop": 0.0,
        "embed_dim": 1024,
        "enc_embed_dim": 1024,
        "mlp_ratio": 4.0,
        "num_heads": 16,
        "qkv_bias": True,
        "random_image_idx_embedding": True,
        "number_ref_views": 2,
    }
}

def resize_pos_embed(pos_embed, old_grid_size, new_grid_size):
    # Separate cls token and patch tokens
    cls_token = pos_embed[:, :1, :]  # [1, 1, C]
    patch_tokens = pos_embed[:, 1:, :]  # [1, N, C]
    
    C = patch_tokens.shape[-1]
    patch_tokens = patch_tokens.reshape(1, old_grid_size[0], old_grid_size[1], C).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(patch_tokens, size=new_grid_size, mode='bicubic', align_corners=False)
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, C)
    
    return torch.cat((cls_token, patch_tokens), dim=1)

@dataclass
class BackboneVggtCfg:
    name : Literal["vggt"]

    # This is all useless
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt
class BackboneVggt(Aggregator):
    def __init__(self, cfg: BackboneVggtCfg, d_in: int) -> None:
        super().__init__(img_size=252)

        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        ckpt_weights = torch.hub.load_state_dict_from_url(_URL)
        ckpt_weights = {k[len('aggregator.'):]: v for k, v in ckpt_weights.items() if k.startswith('aggregator.')}
        ckpt_weights['patch_embed.pos_embed'] = resize_pos_embed(
            ckpt_weights['patch_embed.pos_embed'],
            (37, 37),
            (18, 18)
        )
        print("[backbone_vggt.py]NOT LOADING ORIGINAL VGGT WEIGHTS")
        # missing_keys, unexpected_keys = self.load_state_dict(ckpt_weights, strict=True)

        # self.patch_embed_cls = cfg.patch_embed_cls
        # self.intrinsics_embed_encoder_dim = 0
        # self._set_patch_embed(fast3r_params['encoder_args']['img_size'], fast3r_params['encoder_args']['patch_size'], fast3r_params['encoder_args']['embed_dim'])
        # self.dec_depth = fast3r_params['decoder_args']['depth']
        # self.enc_embed_dim = fast3r_params['encoder_args']['embed_dim']
        # self.dec_embed_dim = fast3r_params['decoder_args']['embed_dim']
        self.conf_mode = ('exp', 1, inf)
        self.dec_depth = 24
        self.enc_embed_dim = 1024
        self.dec_embed_dim = 1024
    """
    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        #in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = self.get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim) #Modify to handle intrinsics information
   """ 

