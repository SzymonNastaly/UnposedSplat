from .fast3r.fast3r import Fast3R
from dataclasses import dataclass
from typing import Literal

fast3r_params = {
    'encoder_args' : {
        'encoder_type': 'croco',
        'img_size': 512,
        'patch_size': 16,
        'patch_embed_cls': 'ManyAR_PatchEmbed',
        'embed_dim': 1024,
        'num_heads': 16,
        'depth': 24,
        'mlp_ratio': 4,
        'pos_embed': 'RoPE100',
        'attn_implementation': 'flash_attention',
    },
    'decoder_args' : {
        'decoder_type': 'fast3r',
        'random_image_idx_embedding': True,
        'enc_embed_dim': 1024, # same as encoder embed_dim
        'embed_dim': 768,
        'num_heads': 12,
        'depth': 12,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop': 0.0,
        'attn_drop': 0.0,
        'attn_implementation': 'flash_attention',
    }
}

@dataclass
class BackboneFast3RCfg:
    name : Literal["fast3r"]
    model: Literal["fast3r"]
    d_out: int

class BackboneFast3R(Fast3R):
    def __init__(self, cfg: BackboneFast3RCfg, d_in: int) -> None:
        super().__init__(fast3r_params['encoder_args'], fast3r_params['decoder_args'])
        