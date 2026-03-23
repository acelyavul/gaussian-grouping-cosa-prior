import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundedSlotDictionary(nn.Module):
    def __init__(
        self,
        num_entries: int = 64,
        slot_dim: int = 16,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_entries = num_entries
        self.slot_dim = slot_dim
        self.temperature = temperature

        # canonical object/property vectors
        self.codebook = nn.Parameter(torch.randn(num_entries, slot_dim) * 0.02)

        # Gaussian prior params for conditional slot init
        self.slot_mu = nn.Parameter(torch.randn(num_entries, slot_dim) * 0.02)
        self.slot_logsigma = nn.Parameter(torch.zeros(num_entries, slot_dim))

    def soft_lookup(self, object_feats: torch.Tensor):
        """
        object_feats: [B, K, D]
        returns:
            dict_embed: [B, K, D]
            weights: [B, K, M]
            indices: [B, K]
        """
        object_feats = F.normalize(object_feats, dim=-1)
        codebook = F.normalize(self.codebook, dim=-1)

        logits = torch.einsum("bkd,md->bkm", object_feats, codebook) / self.temperature
        weights = F.softmax(logits, dim=-1)
        dict_embed = torch.einsum("bkm,md->bkd", weights, self.codebook)
        indices = weights.argmax(dim=-1)
        return dict_embed, weights, indices

    def sample_conditional_slots(self, weights: torch.Tensor, num_slots: int = None):
        """
        weights: [B, K, M]
        returns init slots [B, K, D]
        """
        mu = torch.einsum("bkm,md->bkd", weights, self.slot_mu)
        sigma = torch.exp(torch.einsum("bkm,md->bkd", weights, self.slot_logsigma))
        eps = torch.randn_like(mu)
        slots = mu + sigma * eps
        return slots

    def commitment_loss(self, object_feats: torch.Tensor, dict_embed: torch.Tensor):
        return F.mse_loss(object_feats, dict_embed.detach()) + 0.25 * F.mse_loss(dict_embed, object_feats.detach())

    def usage_regularization(self, weights: torch.Tensor):
        """
        weights: [B, K, M]
        encourage non-collapsed dictionary usage
        """
        avg_usage = weights.mean(dim=(0, 1))  # [M]
        uniform = torch.full_like(avg_usage, 1.0 / avg_usage.numel())
        return F.kl_div(avg_usage.clamp_min(1e-8).log(), uniform, reduction="batchmean")