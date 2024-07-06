"""Attention layer with > 1 current sequence length."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import xformers.ops as xops

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)

class SeqAttnBackend(AttentionBackend):
    
    @staticmethod
    def get_name() -> str:
        return "seq_attn"
    
    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return SeqAttnImpl
    
    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return SeqAttnMetadata

@dataclass
class SeqAttnMetadata:
    seq_lens: List[int]
    past_seq_lens: List[int]

'''
The KVCache
'''

class SeqAttnImpl(AttentionImpl[AttentionMetadata]):
    def __init__(
        self, 
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        assert alibi_slopes is None, "Alibi Slopes not supported for SeqAttn"
        assert sliding_window is None, "Sliding Window not supported for SeqAttn"
        assert blocksparse_params is None, "Blocksparse not supported for SeqAttn"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: "SeqAttnBackend",
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        # query: torch.tensor [Sum(SeqLens), NumHeads * HeadSize]
        # key: torch.tensor [Sum(SeqLens), NumHeads * HeadSize]
        # value: [Sum(SeqLens), NumHeads * HeadSize]
        # kv_cache: Tuple[[2, MaxSeqLen_i, NumHeads, HeadSize], ...]
        assert kv_scale == 1.0, "KV Scale not supported for SeqAttn"
        # Split the query, key, and value into the individual sequences
        # print('attn metadata', attn_metadata)
        accum_seq_len = 0
        outputs = []
        for cache, past_seq_len, seq_len in zip(kv_cache, attn_metadata.past_seq_lens, attn_metadata.seq_lens):  
            q = query[accum_seq_len:accum_seq_len + seq_len].view(1, seq_len, self.num_heads, self.head_size)
            cache[0][past_seq_len:past_seq_len + seq_len] = key[accum_seq_len:accum_seq_len + seq_len]
            cache[1][past_seq_len:past_seq_len + seq_len] = value[accum_seq_len:accum_seq_len + seq_len]
            accum_seq_len += seq_len
            k = cache[0][:past_seq_len + seq_len].view(1, seq_len + past_seq_len, self.num_heads, self.head_size)
            v = cache[1][:past_seq_len + seq_len].view(1, seq_len + past_seq_len, self.num_heads, self.head_size)
            output = xops.memory_efficient_attention_forward(q, k, v, scale = self.scale, attn_bias=xops.fmha.attn_bias.LowerTriangularFromBottomRightMask())
            outputs.append(output.view(seq_len, -1))
        return torch.cat(outputs, dim=0)