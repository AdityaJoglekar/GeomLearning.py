#
import math
import torch
from torch import nn
# from torch.nn import functional as F
# from timm.layers import trunc_normal_, Mlp
# from einops import rearrange, repeat

from am.models.upt.models.upt import UPT
from am.models.upt.models.approximator import Approximator
from am.models.upt.models.decoder_perceiver import DecoderPerceiver
from am.models.upt.models.encoder_supernodes import EncoderSupernodes
from am.models.upt.models.conditioner_timestep import ConditionerTimestep


__all__ = [
    "UPT_t",
]

# # the incremental speedup isn't worth dealing with versioning hell
# FastGELU = lambda: nn.GELU(approximate='tanh')
FastGELU = nn.GELU

def modulate(x, shift, scale):
    '''
    Modulate the input x with shift and scale with
    AdaLayerNorm (DiT, https://arxiv.org/abs/2302.07459)
    '''
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#======================================================================#
# Embeddings
#======================================================================#
class TimeEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.

        Ref: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class UPT_t(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        data,
        n_layers=5,
        n_hidden=128,
        n_head=8,
        mlp_ratio=1,
        num_latent=32
    ):
        super().__init__()

        # positions need to be separated from input x. Following code input_dim etc. is for cylinder flow dataset
        #batch_idx=0


        # self.conditioner = ConditionerTimestep(
        #     dim=n_hidden,
        #     num_timesteps=data.get(0).metadata['num_steps'])

        self.t_embedding = TimeEmbedding(n_hidden)
    
        self.encoder=EncoderSupernodes(
                # simulation has 9 inputs: node values one hot vector +  2D velocity
                input_dim=9,
                # 2D dataset
                ndim=2,
                # positions are rescaled to [0, 200]
                radius=0.05,
                # in regions where there are a lot of mesh cells, it would result in supernodes having a lot of
                # connections to nodes. but since we sample the supernodes uniform, we also have a lot of supernodes
                # in dense regions, so we can simply limit the maximum amount of connections to each supernodes
                # to avoid an extreme amount of edges
                max_degree=32,
                # dimension for the supernode pooling -> use same as ViT-T latent dim
                gnn_dim=n_hidden,
                # ViT-T latent dimension
                enc_dim=n_hidden,
                enc_num_heads=n_head,
                # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
                enc_depth=n_layers//3,
                # downsample to 128 latent tokens for fast training
                perc_dim=n_hidden,
                perc_num_heads=n_head,
                num_latent_tokens=num_latent,
                # pass conditioner dim
                # cond_dim=self.conditioner.cond_dim,
                cond_dim =n_hidden
            )
        self.approximator=Approximator(
                # tell the approximator the dimension of the input (perc_dim or enc_dim of encoder)
                input_dim=n_hidden,
                # as in ViT-T
                dim=n_hidden,
                num_heads=n_head,
                # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
                depth=n_layers//3,
                # pass conditioner dim
                # cond_dim=self.conditioner.cond_dim,
                cond_dim =n_hidden
            )
        self.decoder=DecoderPerceiver(
                # tell the decoder the dimension of the input (dim of approximator)
                input_dim=n_hidden,
                # 2D velocity + pressure
                output_dim=out_dim,
                # simulation is 2D
                ndim=2,
                # as in ViT-T
                dim=n_hidden,
                num_heads=n_head,
                # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
                depth=n_layers//3,
                # we assume num_outputs to be constant so we can simply reshape the dense result into a sparse tensor
                unbatch_mode="dense_to_sparse_unpadded",
                # pass conditioner dim
                # cond_dim=self.conditioner.cond_dim,
                cond_dim =n_hidden
            )

    def forward(self, data):

        if data.get('t_val', None) is not None:
            t = data.t_val.item()
        elif data.get('t', None) is not None:
            t = data.t[0].item()
        else:
            raise RuntimeError(f't_val, t is None in {data}')
        
        t = torch.tensor([t], dtype=data.x.dtype, device=data.x.device) # [B=1]
        c = self.t_embedding(t) # [B=1, C]
        
        if data.get('dt_val', None) is not None:
            d = data.dt_val.item()
            d = torch.tensor([d], dtype=data.x.dtype, device=data.x.device) # [B=1]
            d = self.d_embedding(d)
            c = c + d

        input_feat = torch.cat((data.x[:,0:7],data.x[:,9:]),dim = 1)
        input_pos = data.x[:,7:9]
        supernodes_idxs = data.supernodes_idxs
        output_pos = data.x[:,7:9].unsqueeze(0)
        batch_idx = torch.zeros((data.x.shape[0]), dtype=data.x.dtype, device=data.x.device)
        # timestep = t
        # condition = self.conditioner(timestep)
        condition = c

        # encode data
        latent = self.encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernodes_idxs,
            batch_idx=batch_idx,
            condition=condition,
        )

        # propagate forward
        latent = self.approximator(latent, condition=condition)

        # decode
        pred = self.decoder(
            x=latent,
            output_pos=output_pos,
            condition=condition,
        )

        return pred # [N, C]
