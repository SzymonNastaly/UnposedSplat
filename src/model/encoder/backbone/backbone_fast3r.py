from .fast3r.fast3r import Fast3R
from dataclasses import dataclass
from typing import Literal
from .fast3r.patch_embed import get_patch_embed

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
        self._set_patch_embed(fast3r_params['encoder_args']['img_size'], fast3r_params['encoder_args']['patch_size'], fast3r_params['encoder_args']['embed_dim'])
        self.dec_depth = fast3r_params['decoder_args']['depth']
        self.enc_embed_dim = fast3r_params['encoder_args']['embed_dim']
        self.dec_embed_dim = fast3r_params['decoder_args']['embed_dim']
        self.conf_mode = ('exp', 1, inf)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        #in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim) #Modify to handle intrinsics information

    
        