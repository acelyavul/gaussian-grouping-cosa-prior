import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_attention import SlotAttention
from .grounded_slot_dict import GroundedSlotDictionary


class CoSAPrior(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        slot_dim: int = 16,
        num_slots: int = 8,
        num_dict_entries: int = 64,
        iters: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.pre_norm = nn.LayerNorm(slot_dim)

        self.gsd = GroundedSlotDictionary(
            num_entries=num_dict_entries,
            slot_dim=slot_dim,
        )

        self.slot_attention = SlotAttention(
            num_iterations=iters,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            hidden_dim=slot_dim * 2,
        )

    def forward(self, features: torch.Tensor):
        """
        features: [B, N, input_dim]
        """
        x = self.input_proj(features)         # [B, N, Ds]
        x = self.pre_norm(x)

        # simple token pooling -> pseudo object seeds
        pooled = self._init_object_tokens(x, self.num_slots)  # [B, K, Ds]

        dict_embed, weights, indices = self.gsd.soft_lookup(pooled)
        init_slots = self.gsd.sample_conditional_slots(weights)

        refined_slots, attn = self.slot_attention(x, init_slots)

        aux = {
            "pooled_tokens": pooled,
            "dict_embed": dict_embed,
            "dict_weights": weights,
            "dict_indices": indices,
            "init_slots": init_slots,
            "attn": attn,
        }
        return refined_slots, aux

    def compute_losses(self, aux):
        pooled = aux["pooled_tokens"]
        dict_embed = aux["dict_embed"]
        weights = aux["dict_weights"]

        loss_commit = self.gsd.commitment_loss(pooled, dict_embed)
        loss_usage = self.gsd.usage_regularization(weights)

        return {
            "cosa_commitment": loss_commit,
            "cosa_usage": loss_usage,
        }

    def _init_object_tokens(self, x: torch.Tensor, num_slots: int):
        """
        Cheap initialization by chunk pooling.
        x: [B, N, D]
        """
        B, N, D = x.shape
        if N < num_slots:
            pad = num_slots - N
            x = torch.cat([x, x[:, :pad]], dim=1)
            N = x.shape[1]

        chunks = torch.chunk(x, num_slots, dim=1)
        pooled = [c.mean(dim=1) for c in chunks]
        return torch.stack(pooled, dim=1)