import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_iterations: int = 3,
        slot_dim: int = 16,
        input_dim: int = 256,
        hidden_dim: int = 128,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.slot_dim = slot_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.scale = slot_dim ** -0.5

    def forward(self, inputs: torch.Tensor, init_slots: torch.Tensor):
        """
        inputs: [B, N, D_in]
        init_slots: [B, K, D_slot]
        """
        B, N, _ = inputs.shape
        K = init_slots.shape[1]

        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)   # [B, N, Ds]
        v = self.to_v(inputs)   # [B, N, Ds]

        slots = init_slots

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            q = self.to_q(slots_norm)  # [B, K, Ds]
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1) + self.eps   # normalize over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)      # stabilize over tokens

            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            slots = self.gru(
                updates.reshape(B * K, -1),
                slots_prev.reshape(B * K, -1),
            ).reshape(B, K, -1)

            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn