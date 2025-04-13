# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the FAIR Noncommercial Research License
# that can be found at:
# https://github.com/facebookresearch/fast3r/blob/main/LICENSE
#
# This file contains modified portions of the original source code released by Meta
# under the FAIR Noncommercial Research License. It is used here for noncommercial
# academic research purposes only, in accordance with the license terms.
#
# If you distribute this file or any derivatives, you must include the license above
# and comply with all terms of the FAIR Noncommercial Research License.

import os
from copy import deepcopy
import time
from typing import Optional
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from .croco.models.blocks import Block, PositionGetter
from .croco.models.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from .components.llama import TransformerBlock, RMSNorm, precompute_freqs_cis
from .patch_embed import get_patch_embed
from functools import partial



# log = pylogger.RankedLogger(__name__, rank_zero_only=True) # commented out for initial testing

class Fast3R(nn.Module):
    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict
    ):
        super(Fast3R, self).__init__()

        self.encoder_args = OmegaConf.to_container(encoder_args) if isinstance(encoder_args, DictConfig) else encoder_args
        self.build_encoder(encoder_args)

        self.decoder_args = OmegaConf.to_container(decoder_args) if isinstance(decoder_args, DictConfig) else decoder_args
        self.build_decoder(decoder_args)

    def build_encoder(self, encoder_args: dict):
        # Initialize the encoder based on the encoder type
        if encoder_args["encoder_type"] == "croco":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = CroCoEncoder(**encoder_args)
        elif encoder_args["encoder_type"] == "dino_v2":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = DinoEncoder(**encoder_args)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_args['encoder_type']}")

    def build_decoder(self, decoder_args: dict):
        decoder_args["decoder_type"] = decoder_args.get('decoder_type', 'fast3r')  # default to fast3r if not specified
        if decoder_args["decoder_type"] == 'fast3r':
            decoder_args = deepcopy(decoder_args)
            decoder_args.pop('decoder_type')
            self.decoder = Fast3RDecoder(**decoder_args)
        elif decoder_args["decoder_type"] == 'llama':
            decoder_args = deepcopy(decoder_args)
            decoder_args.pop('decoder_type')
            self.decoder = LlamaDecoder(**decoder_args)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_args['decoder_type']}")

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)


    def load_from_dust3r_checkpoint(self, dust3r_checkpoint_path: str):
        """Load a Dust3R checkpoint into the model.
        Only load the patch_embed, enc_blocks, enc_norm components from the checkpoint.

        Args:
            dust3r_checkpoint_path (str): Path to the Dust3R checkpoint.
        """
        # Load the checkpoint
        checkpoint = torch.load(dust3r_checkpoint_path, weights_only=False)['model']

        # Initialize state dictionaries for different components
        encoder_state_dict = {}

        # Prepare to track loaded keys
        loaded_keys = set()

        # Split the checkpoint into encoder and downstream head
        for key, value in checkpoint.items():
            if key.startswith("patch_embed") or key.startswith("enc_blocks") or key.startswith("enc_norm"):
                if isinstance(self.encoder, CroCoEncoder):
                    new_key = key.replace("patch_embed", "encoder.patch_embed") \
                                 .replace("enc_blocks", "encoder.enc_blocks") \
                                 .replace("enc_norm", "encoder.enc_norm")
                    encoder_state_dict[new_key] = value
                    loaded_keys.add(key)  # Tentatively mark as loaded

        # Load the encoder part into the model if it is an instance of CroCoEncoder
        if isinstance(self.encoder, CroCoEncoder):
            load_result = self.load_state_dict(encoder_state_dict, strict=False)

            # Remove keys that failed to load
            missing_keys = set(load_result.missing_keys)
            unexpected_keys = set(load_result.unexpected_keys)
            loaded_keys -= (missing_keys | unexpected_keys)
        del checkpoint


    def _encode_images(self, views, chunk_size=400):
        B = views[0]["img"].shape[0]

        # Check if all images have the same shape
        same_shape = all(view["img"].shape == views[0]["img"].shape for view in views)

        if same_shape:
            # Stack images along a new dimension to create a batch
            imgs = torch.cat([view["img"] for view in views], dim=0)  # Shape: [num_views * B, C, H, W]
            true_shapes = torch.cat(
                [view.get("true_shape", torch.tensor(view["img"].shape[-2:])[None].repeat(B, 1)) for view in views],
                dim=0
            )  # Shape: [num_views * B, 2]

            # Encode images in chunks to prevent OOM
            num_chunks = (imgs.shape[0] + chunk_size - 1) // chunk_size
            feats_chunks = []
            pos_chunks = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, imgs.shape[0])
                chunk_feats, chunk_pos = self.encoder(imgs[start_idx:end_idx], true_shapes[start_idx:end_idx])
                feats_chunks.append(chunk_feats)
                pos_chunks.append(chunk_pos)
            
            feats = torch.cat(feats_chunks, dim=0)
            pos = torch.cat(pos_chunks, dim=0)

            # Split the encoded features and positions back into individual views
            encoded_feats = torch.split(feats, B, dim=0)
            positions = torch.split(pos, B, dim=0)
            shapes = torch.split(true_shapes, B, dim=0)
        else:
            # Process each image individually
            encoded_feats, positions, shapes = [], [], []
            for view in views:
                img = view["img"]
                true_shape = view.get(
                    "true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1)
                )
                feat, pos = self.encoder(img, true_shape)
                encoded_feats.append(feat)
                positions.append(pos)
                shapes.append(true_shape)

        return encoded_feats, positions, shapes

    def forward(self, context):
        """
        Args:
            context dict: contains the input data for the model, including images and their true shapes.

        Returns:
            dec_feat: List of decoded image features from the decoder.
            shape: Tensor containing the height and width of the images.
            images_all: Tensor containing all input images.
        """
        images_all = context["image"]
        b, v, _, h, w = images_all.shape
        image_batches = torch.unbind(images_all, dim=1)
        views = [{"img": i} for i in image_batches]
        
        # encode the images --> B,S,D
        encoded_feats, positions, shapes = self._encode_images(views)

        # Create image IDs for each patch
        num_images = len(views)
        B, _, _ = encoded_feats[0].shape

        # Initialize an empty list to collect image IDs for each patch.
        # Note that at inference time, different views may have different number of patches.
        image_ids = []

        # Loop through each encoded feature to get the actual number of patches
        for i, encoded_feat in enumerate(encoded_feats):
            num_patches = encoded_feat.shape[1]  # Get the number of patches for this image
            # Extend the image_ids list with the current image ID repeated num_patches times
            image_ids.extend([i] * num_patches)

        # Repeat the image_ids list B times and reshape it to match the expected shape
        image_ids = torch.tensor(image_ids * B).reshape(B, -1).to(encoded_feats[0].device)

        dec_output = self.decoder(encoded_feats, positions, image_ids)
        
        # Assume all images have the same number of patches
        P_patches = encoded_feats[0].shape[1]
        dec_feat = []
        for layer_output in dec_output:
            # layer_output: (B, num_images * P_patches, D)
            # Rearrange to (B, num_images, P_patches, D)
            layer_output = rearrange(
                layer_output,
                'B (num_images P_patches) D -> B num_images P_patches D',
                num_images=num_images,
                P_patches=P_patches
            )
            dec_feat.append(layer_output)
        shape = torch.tensor([h, w])[None, None, :].repeat(b, v, 1)
        return dec_feat, shape, images_all

class CroCoEncoder(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        patch_embed_cls="ManyAR_PatchEmbed",
        embed_dim=768,
        num_heads=12,
        depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_embed="RoPE100",
        attn_implementation="pytorch_naive",
    ):
        super(CroCoEncoder, self).__init__()

        # patch embeddings  (with initialization done as in MAE)
        self.patch_embed_cls = patch_embed_cls
        self._set_patch_embed(img_size, patch_size, embed_dim)

        # Positional embedding
        self.pos_embed = pos_embed
        if pos_embed.startswith("RoPE"):  # eg RoPE100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        # Transformer blocks
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer,
                  rope=self.rope,
                  attn_implementation=attn_implementation)
            for _ in range(depth)
        ])
        self.enc_norm = norm_layer(embed_dim)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def forward(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # Apply encoder blocks
        for blk in self.enc_blocks:
            x = blk(x, pos)

        # Apply final normalization
        x = self.enc_norm(x)
        return x, pos

class DinoEncoder(nn.Module):
    def __init__(
        self,
        patch_size=14,
        **kwargs
    ):
        super(DinoEncoder, self).__init__()
        # Load the pretrained DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        assert self.model.patch_size == patch_size == 14, "DINOv2 model must have patch size 14"
        self.patch_size = patch_size
        self.position_getter = PositionGetter()

    def forward(self, image, true_shape):
        # image shape: B x C x H x W
        B, C, H, W = image.shape

        # Split the batch into landscape and portrait based on true_shape
        landscape_mask = true_shape[:, 1] >= true_shape[:, 0]  # width >= height (landscape)
        portrait_mask = ~landscape_mask  # width < height (portrait)

        # Calculate the number of patches for the largest resolution in the batch
        true_height = true_shape[:, 0]  # Index 0 is height
        true_width = true_shape[:, 1]   # Index 1 is width
        num_patches_h = true_height // self.patch_size
        num_patches_w = true_width // self.patch_size
        num_patches = num_patches_h * num_patches_w  # Total number of patches

        # Pre-allocate tensors for the output
        encoded_feats = torch.empty((B, num_patches.max(), self.model.embed_dim), dtype=next(self.named_parameters())[1].dtype, device=image.device)
        encoded_pos = torch.empty((B, num_patches.max(), 2), dtype=torch.long, device=image.device)

        # If there are landscape images, process them
        if landscape_mask.any():
            landscape_images = image[landscape_mask]
            landscape_shapes = true_shape[landscape_mask]
            landscape_features, landscape_pos = self._process_images(landscape_images, landscape_shapes)
            encoded_feats[landscape_mask] = landscape_features
            encoded_pos[landscape_mask] = landscape_pos

        # If there are portrait images, process them
        if portrait_mask.any():
            portrait_images = image[portrait_mask]
            portrait_shapes = true_shape[portrait_mask]

            # Transpose the portrait images back to their original orientation
            portrait_images_transposed = portrait_images.transpose(2, 3)  # HxW -> WxH
            portrait_features, portrait_pos = self._process_images(portrait_images_transposed, portrait_shapes)

            # Unflatten the features, transpose back to match original batch order, then flatten again
            num_patches_h = portrait_shapes[:, 0] // self.patch_size  # Use true height
            num_patches_w = portrait_shapes[:, 1] // self.patch_size  # Use true width
            B_p, N, D = portrait_features.shape

            # Unflatten the features to (B, num_patches_h, num_patches_w, D)
            portrait_features_unflattened = portrait_features.view(B_p, num_patches_h[0], num_patches_w[0], D)

            # Transpose back (swap height and width)
            portrait_features_transposed = portrait_features_unflattened.transpose(1, 2)

            # Flatten again to match the expected shape
            portrait_features_flattened = portrait_features_transposed.flatten(1, 2)

            # Apply the same operation for positional embeddings (pos)
            B_p, N, _ = portrait_pos.shape  # Get the shape for pos
            portrait_pos_unflattened = portrait_pos.view(B_p, num_patches_h[0], num_patches_w[0], 2)
            portrait_pos_transposed = portrait_pos_unflattened.transpose(1, 2)
            portrait_pos_flattened = portrait_pos_transposed.flatten(1, 2)

            # Assign the processed features and positional embeddings back
            encoded_feats[portrait_mask] = portrait_features_flattened
            encoded_pos[portrait_mask] = portrait_pos_flattened

        return encoded_feats, encoded_pos

    def _process_images(self, images, true_shape):
        """
        Process a batch of images through the DINO encoder and compute positions.
        """
        # Forward pass through the DINO encoder to get encoded features
        features = self.model.forward_features(images)['x_norm_patchtokens']  # Shape: B x N_patches x D
        x = features  # Encoded features

        # Compute positions using PositionGetter
        true_height = true_shape[:, 0]  # Explicitly assign height
        true_width = true_shape[:, 1]   # Explicitly assign width
        num_patches_h = true_height // self.patch_size  # Height patches
        num_patches_w = true_width // self.patch_size  # Width patches
        pos = self.position_getter(images.shape[0], num_patches_h[0], num_patches_w[0], images.device)

        return x, pos


class Fast3RDecoder(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_implementation: str = "pytorch_naive",
        attn_bias_for_inference_enabled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(Fast3RDecoder, self).__init__()

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        self.dec_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=nn.LayerNorm,
                attn_implementation=attn_implementation,
                attn_bias_for_inference_enabled=attn_bias_for_inference_enabled
            ) for _ in range(depth)
        ])

        # initialize the positional embedding for the decoder
        self.random_image_idx_embedding = random_image_idx_embedding
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        # final norm layer
        self.dec_norm = norm_layer(embed_dim)

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but deterministic given a fixed number of forward passes (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random non-repeating IDs for the remaining views using the generator
        for b in range(batch_size):
            # Use the torch.Generator for randomness to ensure randomness between forward passes
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def forward(self, encoded_feats, positions, image_ids):
        """ Forward pass through the decoder.

        Args:
            encoded_feats (list of tensors): Encoded features for each view. Shape: B x Npatches x D
            positions (list of tensors): Positional embeddings for each view. Shape: B x Npatches x 2
            image_ids (tensor): Image IDs for each patch. Shape: B x Npatches
        """
        x = torch.cat(encoded_feats, dim=1)  # concate along the patch dimension
        pos = torch.cat(positions, dim=1)

        final_output = [x]  # before projection

        # project to decoder dim
        x = self.decoder_embed(x)

        # Add positional embedding based on image IDs
        if self.random_image_idx_embedding:
            # Generate random positional embeddings for all views and samples
            image_pos = self._get_random_image_pos(encoded_feats=encoded_feats,
                                                   batch_size=encoded_feats[0].shape[0],
                                                   num_views=len(encoded_feats),
                                                   max_image_idx=self.image_idx_emb.shape[0] - 1,
                                                   device=x.device)
        else:
            # Use default image IDs from input
            num_images = (torch.max(image_ids) + 1).cpu().item()
            image_idx_emb = self.image_idx_emb[:num_images]
            image_pos = image_idx_emb[image_ids]

        # Apply positional embedding based on image IDs and positions
        x += image_pos  # x has size B x Npatches x D, image_pos has size Npatches x D, so this is broadcasting

        for blk in self.dec_blocks:
            x = blk(x, pos)
            final_output.append(x)

        x = self.dec_norm(x)
        final_output[-1] = x

        return final_output

class LlamaDecoder(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000,
        max_seq_len: int = 1000,
        is_causal: bool = False,  # use bidirectional attention
        depth_init: bool = True,
        **kwargs
    ):
        super(LlamaDecoder, self).__init__()

        # assign the flags to attributes for later use
        self.random_image_idx_embedding = random_image_idx_embedding
        self.rope_theta = rope_theta

        # Compute head dimension
        self.head_dim = embed_dim // n_heads

        # Precompute freqs_cis
        self.precomputed_freqs_cis = self._precompute_freqs_cis(max_seq_len=max_seq_len)  # complex64, it is a tensor and not a parameter or buffer because otherwise DeepSpeed will convert it to float32

        # **Learnable embedding for view 0**
        self.view0_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.view0_embed, mean=0.0, std=0.02)

        # Transfer from encoder to decoder dimensions
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        # Initialize Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, n_heads=n_heads, n_kv_heads=n_kv_heads, dim=embed_dim, multiple_of=multiple_of,
                             ffn_dim_multiplier=ffn_dim_multiplier, n_layers=n_layers, is_causal=is_causal, norm_eps=norm_eps, depth_init=depth_init)
            for i in range(n_layers)
        ])

        self.norm = RMSNorm(dim=embed_dim, eps=norm_eps)

    def _precompute_freqs_cis(self, max_seq_len) -> torch.Tensor:
        return precompute_freqs_cis(
            self.head_dim,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            max_seq_len,
            self.rope_theta,
        )

    def _generate_per_rank_generator(self):
        # Generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_freqs_cis(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates freqs_cis for each patch based on random image indices.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            torch.Tensor: freqs_cis of shape (batch_size, total_num_patches, head_dim)
        """
        # Generate random image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random IDs for the remaining views
        for b in range(batch_size):
            # Use the torch.Generator for randomness
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)
        self.precomputed_freqs_cis = self.precomputed_freqs_cis.to(device)

        # Initialize list to store positional embeddings for all views
        freqs_cis_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            freqs_cis_for_view = self.precomputed_freqs_cis[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            freqs_cis_for_view = freqs_cis_for_view.unsqueeze(1).repeat(1, num_patches, 1)  # (B, Npatches, D)

            freqs_cis_list.append(freqs_cis_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        freqs_cis = torch.cat(freqs_cis_list, dim=1)  # (B, Npatches_total, D)

        return freqs_cis

    def forward(self, encoded_feats, positions, image_ids):
        x = torch.cat(encoded_feats, dim=1)  # Concatenate along the patch dimension
        pos = torch.cat(positions, dim=1)
        batch_size = x.shape[0]
        device = x.device

        x = self.decoder_embed(x)

        # Generate freqs_cis based on image_ids
        if self.random_image_idx_embedding:
            freqs_cis = self._get_random_freqs_cis(
                encoded_feats=encoded_feats,
                batch_size=batch_size,
                num_views=len(encoded_feats),
                max_image_idx=self.precomputed_freqs_cis.shape[0] - 1,
                device=device
            )
        else:
            # Use image_ids to index into precomputed_freqs_cis
            num_images = (torch.max(image_ids) + 1).cpu().item()
            self.precomputed_freqs_cis = self.precomputed_freqs_cis.to(device=device)
            image_idx_emb = self.precomputed_freqs_cis[:num_images]
            freqs_cis = image_idx_emb[image_ids]

        # Create a mask for view 0 patches
        view0_mask = (image_ids == 0).unsqueeze(-1).float()  # Shape: (batch_size, total_num_patches, 1)

        final_output = [x]

        for layer in self.layers:
            # Add the view0_embedding to the features of view 0 before each transformer layer
            x = x + view0_mask * self.view0_embed  # Broadcasts self.view0_embed over the last dimension

            x = layer(x, freqs_cis)
            final_output.append(x)

        x = self.norm(x)
        final_output[-1] = x

        return final_output