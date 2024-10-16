import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import functools
import math
import warnings
import json

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from torch import nn
from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb, repeat_kv

class Gemma:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_activations = {
            (layer, head): None
            for layer in range(len(self.model.model.layers))
            for head in range(self.model.model.layers[0].self_attn.num_heads)
        }
        self.steer = False
        self.break_into()
    
    def set_steering_vectors(self, vectors, positions_to_steer):
        self.steering_vectors = {(pos[0], pos[1]):
                                 (torch.tensor(vectors[i]), pos[2]) for i, pos in enumerate(positions_to_steer)}
        self.steer = True
        
    def break_into(self) -> None:
        self.hook_handles = []
        self.prev_forwards = []

        for layer in range(len(self.model.model.layers)):
            self.prev_forwards.append(self.model.model.layers[layer].self_attn.forward)
            forward_partial = functools.partial(self.attn_forward, layer=layer,
                                                self=self.model.model.layers[layer].self_attn,
                                                gemma_model=self)
            self.model.model.layers[layer].self_attn.forward = forward_partial

    def break_out(self) -> None:
        for layer, f in enumerate(self.prev_forwards):
            forward_partial = functools.partial(self.prev_forwards[layer],
                                                self=self.model.model.layers[layer].self_attn)
            self.model.model.layers[layer].self_attn.forward = forward_partial
        for h in self.hook_handles:
            h.remove()

    @staticmethod
    def attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer: Optional[int] = None,
        gemma_model: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if gemma_model.steer:
            for (cur_layer, cur_head) in gemma_model.steering_vectors:
                if cur_layer == layer:
                    attn_output[:, gemma_model.steering_vectors[(cur_layer, cur_head)][1], cur_head, :] += gemma_model.steering_vectors[(cur_layer, cur_head)][0].to(attn_output.device)
        
        for cur_head in range(self.num_heads):
            gemma_model.cache_activations[(layer, cur_head)] = attn_output[:, :, cur_head, :].detach().cpu().clone()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value