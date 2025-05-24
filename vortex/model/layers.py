# Copyright (c) 2024, Michael Poli.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable # Callable can be removed if not used elsewhere
from vortex.model.utils import grab_first_if_tuple


# try:
#     from hyena_ops import hyena_se_fwd, hyena_mr_fwd, hyena_li_fwd
# except ImportError:
#     hyena_se_fwd, hyena_mr_fwd, hyena_li_fwd = None, None, None


class TELinear(nn.Linear):
    """
    Wrapper for torch.nn.Linear layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        # init_method: Callable, # torch.nn.Linear has its own init
        **kwargs,
    ):
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            **kwargs,
        )
        # If init_method is crucial, it might need to be applied manually here
        # For example:
        # if init_method is not None:
        #     init_method(self.weight)
        #     if bias:
        #         # Initialize bias if needed, e.g., to zeros
        #         with torch.no_grad():
        #             self.bias.zero_()


    def forward(self, x):
        out = super().forward(x)
        return out


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size, dtype=config.params_dtype))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = False # Hardcoded to False

        # if self.use_flash_rmsnorm: # This block can be removed
        #     from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func
        #     self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        # if self.use_flash_rmsnorm: # This condition will always be false
        #     return self.rmsnorm_func(x, self.scale, self.eps)
        # else:
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
            return self.scale * y


class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError
        
        if self.layer_idx > 0 and config.get("evo2_style_activations", False):
            self.act = nn.Identity()

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        inner_size = config.get("inner_mlp_size", inner_size)

        self.l1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=config.hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class Embedding(nn.Module):
    _train_dtype = "bf16"

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

    def embed(self, input_ids, position_ids=None, tokentype_ids=None):
        embeddings = self.word_embeddings(input_ids)
        return embeddings

    def unembed(self, u):
        weight = self.word_embeddings.weight
        return torch.matmul(u, weight)


class VocabParallelEmbedding(nn.Embedding):
    "Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py"

    def __init__(self, config):
        vocab_size, padding_idx = (
            config.vocab_size,
            config.get("padding_idx", None),
        )
        # self.process_group = None # Removed
        super().__init__(
            vocab_size, # Use full vocab_size
            embedding_dim=config.hidden_size,
            padding_idx=padding_idx,
        )

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def unembed(self, u: Tensor) -> Tensor:
        return u @ self.weight.T


class VocabParallelUnembedding(VocabParallelEmbedding):
    def forward(self, input: Tensor) -> Tensor:
        return self.unembed(input)


# class HyenaSE(nn.Module):
#     def __init__(
#             self,
#             hidden_size,
#             num_filters,
#             l_max,
#             order=2,
#             filter_order=64,
#             num_heads=1,
#             inner_factor=1,
#             num_blocks=1,
#             fused_bias_fc=False,
#             outer_mixing=False,
#             dropout=0.0,
#             filter_dropout=0.0,
#             filter_cls='hyena-filter',
#             post_order_ffn=False,
#             jit_filter=False,
#             short_filter_order=3,
#             activation="id",
#             return_state=False,
#             **filter_args,
#         ):
#         r"""
#         Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

#         Args:
#             d_model (int): Dimension of the input and output embeddings (width of the layer)
#             l_max: (int): Maximum input sequence length. Defaults to None
#             order: (int): Depth of the Hyena recurrence. Defaults to 2
#             filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
#             num_heads: (int): Number of heads. Defaults to 1
#             inner_factor: (int): Width multiplier. Defaults to 1
#             num_blocks: (int): Number of blocks in sequence length. Defaults to 1
#             fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
#             dropout: (float): Dropout probability. Defaults to 0.0
#             filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
#             post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
#             jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
#             short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
#             activation: (str): type of act between kernel output and FF (default identity)
#             return_state: (bool): whether to return a state
#         """
#         super().__init__()
#         assert d_model % num_heads == 0, f'Model dimension {d_model} must be divisible by num heads {num_heads}'
#         assert l_max % num_blocks == 0, f'Maximum signal length {l_max} must be divisible by block dimension {num_blocks}'
#         block_dim = l_max // num_blocks
#         head_dim = d_model // num_heads

#         auto_assign_attrs(
#             self, d_model=d_model, order=order, l_max=l_max, num_heads=num_heads, inner_factor=inner_factor,
#             block_dim=block_dim, head_dim=head_dim, filter_order=filter_order, post_order_ffn=post_order_ffn,
#             short_filter_order=short_filter_order, num_blocks = num_blocks, filter_dropout=filter_dropout,
#             jit_filter=jit_filter, outer_mixing=outer_mixing, activation=activation, return_state=return_state,
#         )
#         self.activation = Activation(activation)
#         self.dropout = nn.Dropout(dropout)
#         self.setup_projections(fused_bias_fc, inner_factor)
#         self.setup_filters(filter_cls, filter_args)


#     def setup_projections(self, fused_bias_fc, inner_factor):
#         "Initializes input and output projections (over the width dimension)"
#         if fused_bias_fc and FusedDense is None:
#             raise ImportError('fused_dense is not installed')
#         linear_cls = nn.Linear if not fused_bias_fc else FusedDense
#         self.out_proj = linear_cls(self.d_model * inner_factor, self.d_model)
#         self.in_proj = linear_cls(self.d_model, (self.order + 1) * self.d_model)
#         if self.post_order_ffn:
#             self.ord_proj_w = nn.Parameter(torch.randn(self.order, self.num_heads, self.num_heads) / math.sqrt(self.head_dim))


#     def setup_filters(self, filter_cls, filter_args):
#         "Initializes the explicit and implicit filters"
#         assert self.order >= 2, f'Order must be at least 2, (got {self.order})'
#         total_width = self.d_model * self.inner_factor * (self.order + 1)

#         # self.short_filter = nn.Conv1d(
#         #     in_channels=total_width,
#         #     out_channels=total_width,
#         #     kernel_size=self.short_filter_order,
#         #     groups=total_width,
#         #     padding=self.short_filter_order - 1
#         # )

#         #print(self.short_filter_order, total_width, 'keshik')
#         self.short_filter = CausalConv1D_Filter(
#             in_channels=total_width,
#             out_channels=total_width,
#             kernel_size=self.short_filter_order,
#             groups=total_width,
#             padding=self.short_filter_order - 1
#         )

#         # filter_cls = instantiate(registry.layer, filter_cls, partial=True)

#         # self.filter_fn = filter_cls(
#         #     self.head_dim * self.inner_factor * (self.order - 1),
#         #     order=self.filter_order,
#         #     seq_len=self.l_max,
#         #     channels=1,
#         #     dropout=self.filter_dropout,
#         #     **filter_args
#         # )
#         # if self.jit_filter: self.filter_fn = torch.jit.script(self.filter_fn, self.L)

#     def recurrence(self, u , state):
#         "Fast inference mode via distilled recurrence"
#         raise NotImplementedError("Working on it!")

#     def forward(self, u, *args, **kwargs):
#         l = u.size(-2)
#         l_filter = min(l, self.l_max)
#         u = self.in_proj(u)
#         u = rearrange(u, 'b l d -> b d l')

#         uc = self.short_filter(u)[...,:l_filter]
#         #print(uc.size())

#         uc = rearrange(uc, 'b (ho v) (z l) -> b ho v z l',
#             z=self.num_blocks,
#             ho=self.num_heads,
#             v=self.head_dim * (self.order + 1)
#         )
#         #print(uc.size())
#         *x, v = uc.split(self.d_model, dim=2)
#         #k = self.filter_fn.filter(l_filter) # not required for short convs

#         # `c` is always 1 by default
#         #k = rearrange(k, 'c l (v o) -> c o v l', v=self.head_dim, o=self.order - 1)[0] # not required for short convs

#         #bias = rearrange(self.filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.order - 1) # not required for short convs

#         #print(self.outer_mixing, self.post_order_ffn, self.activation, self.out_proj, self.dropout  )
#         for o, x_i in enumerate(reversed(x[1:])):
#             if self.outer_mixing:
#                 v = rearrange(v, 'b h v z l -> b h 1 v z l')
#                 v = self.dropout(
#                     v * rearrange(x_i, 'b h v z l -> b h v 1 z l')
#                 )
#                 v = v.sum(dim=2)
#             else:
#                 #print(v[0,0,:])
#                 v = self.dropout(v * x_i)
#                 #rint(v[0,0,:])

#             # the bias term is broadcasted. Last dimension (l) is handled by fftconv
#             #v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None]) # not required for short convs

#             if self.post_order_ffn:
#                 w = self.ord_proj_w[o]
#                 v = mul_sum(
#                     rearrange(w, 'h1 h2 -> 1 h1 h2 1 1 1'), rearrange(v, 'b h v z l -> b h 1 v z l')
#                 )

#         y = self.activation(rearrange(v * x[0], 'b h v z l -> b (z l) (h v)', z=self.num_blocks, h=self.num_heads))
#         y = self.out_proj(y)

#         if self.return_state:
#             return y, None
#         return y

#     @property
#     def d_output(self):
#         return self.d_model


# class ParallelHyenaSE(nn.Module):
#     def __init__(
#         self,
#         hidden_size,
#         global_config,
#         init_method,
#         short_conv_class,
#         use_fast_causal_conv=False,
#         is_mlp=False,
#         local_init=False,
#     ):
#         super().__init__()
#         self.global_config = global_config
#         self.is_mlp = is_mlp
#         self.hidden_size = hidden_size
#         self.use_custom_hyena_mlp_kernel = False
#         self.use_custom_hyena_short_kernel = False
#         self.dense_feat = nn.Linear(hidden_size, hidden_size * 3)
#         self.se_feat = nn.Conv1d(hidden_size, hidden_size, groups=hidden_size, kernel_size=3, padding=1)

#     def featurizer(self, x):
#         pass

#     def forward(self, x):
#         """
#         Note:
#             Input shapes: bs, seq_length, (num_groups, group_size)
#             Output shapes: bs, seq_length, num_groups, group_size
#         """
#         B, L, G, DG = x1.shape
#         x1, x2, v = self.featurizer(x)

#         if self.use_custom_hyena_mlp_kernel or self.use_custom_hyena_short_kernel:
#             z = self.kernel_fn(
#                 x1,
#                 x2,
#                 v,
#                 self.short_conv.short_conv_weight,
#                 repeat_interleave=True,
#                 use_causal_conv=self.use_fast_causal_conv,
#                 autotune=False,
#                 fwd_kernel_cfg=self.fwd_kernel_cfg,
#                 bwd_kernel_cfg=self.bwd_kernel_cfg,
#             )
#             return rearrange(z, "b l g dg -> b l (g dg)", g=G)

#         elif self.use_cgcg_mlp or self.use_cgcg_short:
#             dtype = x1.dtype

#             if self.cgcg_dtype != dtype:
#                 x = v.to(self.cgcg_dtype)
#                 B = x2.to(self.cgcg_dtype)
#                 C = x1.to(self.cgcg_dtype)
#                 h = self.short_conv.short_conv_weight.to(self.cgcg_dtype)  # g, 1, filter_l
#             else:
#                 x = v
#                 B = x2
#                 C = x1
#                 h = self.short_conv.short_conv_weight  # g, 1, filter_l

#             bs, seqlen, g, dg = x.shape

#             z = self.kernel_fn(
#                 x,  # x1.to(self.cgcg_dtype),
#                 B,  # x2.to(self.cgcg_dtype),
#                 C,  # v.to(self.cgcg_dtype),
#                 h,  # g, 1, filter_l
#                 bs=bs,
#                 seqlen=seqlen,
#                 g=g,
#                 dg=dg,
#                 # Explicitly set fwd autotune to False for now
#                 fwd_autotune=False,
#                 bwd_autotune=self.global_config.cgcg_bwd_autotune,
#                 fused_bwd=self.global_config.cgcg_fused_bwd,
#                 fwd_kernel_cfg=self.fwd_kernel_cfg,
#                 bwd_kernel_cfg=None if self.global_config.cgcg_bwd_autotune else self.bwd_kernel_cfg,
#             )
#             out = rearrange(z, "b l g d -> b l (g d)")
#             if self.cgcg_dtype != dtype:
#                 out = out.to(dtype)
#             return out

#         else:
#             x1 = rearrange(x1, "b l g dg -> b (g dg) l")
#             x2 = rearrange(x2, "b l g dg -> b (g dg) l")
#             v = rearrange(v, "b l g dg -> b (g dg) l")
#             x1, x2, v = x1[..., :L], x2[..., :L], v[..., :L]
#             z = x2 * v if self.pregate else v
#             z = self.short_conv(z)
#             z = x1 * z if self.postgate else z
#             return rearrange(z, "b d l -> b l d")


# class HyenaMR(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#     def forward(self, x):
#         raise NotImplementedError

# class HyenaLI(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#     def forward(self, x):
#         raise NotImplementedError
