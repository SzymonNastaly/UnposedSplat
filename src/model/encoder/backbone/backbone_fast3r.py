from .fast3r.fast3r import Fast3R
from dataclasses import dataclass
from typing import Literal
from .fast3r.patch_embed import get_patch_embed

#FIXME: these params should be read from some config file
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

    # This is all useless
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt


class BackboneFast3R(Fast3R):
    def __init__(self, cfg: BackboneFast3RCfg, d_in: int) -> None:
        super().__init__(fast3r_params['encoder_args'], fast3r_params['decoder_args'])
        
        self.patch_embed_cls = cfg.patch_embed_cls
        self.intrinsics_embed_encoder_dim = 0
        self._set_patch_embed()
        self.dec_depth = fast3r_params['decoder_args']['depth']
        self.enc_embed_dim = fast3r_params['encoder_args']['embed_dim']
        self.dec_embed_dim = fast3r_params['decoder_args']['embed_dim']

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        #in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim) #Modify to handle intrinsics information

    
        