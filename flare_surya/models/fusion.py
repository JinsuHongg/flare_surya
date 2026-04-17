import torch
import torch.nn as nn


class FusionStrategy(nn.Module):
    def forward(
        self, img_tokens: torch.Tensor, secondary_tokens: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def requires_pooled(self) -> bool:
        return True  # default to True


class ConcatFusion(FusionStrategy):
    def __init__(self, img_dim: int, secondary_dim: int):
        super().__init__()
        self.img_dim = img_dim
        self.secondary_dim = secondary_dim

    def forward(
        self, img_tokens: torch.Tensor, secondary_tokens: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([img_tokens, secondary_tokens], dim=1)

    @property
    def requires_pooled(self) -> bool:
        return True


class GatedFusion(FusionStrategy):
    def __init__(self, img_dim: int, secondary_dim: int, fuse_dim: int):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, fuse_dim)
        self.secondary_proj = nn.Linear(secondary_dim, fuse_dim)
        self.gate = nn.Linear(fuse_dim, fuse_dim)
        self.fuse_dim = fuse_dim

    def forward(
        self, img_tokens: torch.Tensor, secondary_tokens: torch.Tensor
    ) -> torch.Tensor:
        img_p = self.img_proj(img_tokens)
        secondary_p = self.secondary_proj(secondary_tokens)
        gate = torch.sigmoid(self.gate(secondary_p))
        return gate * img_p + (1 - gate) * secondary_p

    @property
    def requires_pooled(self) -> bool:
        return True


class CrossAttentionFusion(FusionStrategy):
    def __init__(self, img_dim: int, secondary_dim: int, num_heads: int = 8):
        super().__init__()
        self.secondary_proj = nn.Linear(secondary_dim, img_dim)
        self.cross_attn = nn.MultiheadAttention(img_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(img_dim)

    def forward(
        self, img_tokens: torch.Tensor, secondary_tokens: torch.Tensor
    ) -> torch.Tensor:
        secondary_p = self.secondary_proj(secondary_tokens)
        attn_out, _ = self.cross_attn(
            query=secondary_p, key=img_tokens, value=img_tokens
        )
        fused = self.norm(secondary_p + attn_out)
        return fused

    @property
    def requires_pooled(self) -> bool:
        return False


class FusionModule(nn.Module):
    def __init__(
        self,
        fusion_type: str,
        img_dim: int,
        secondary_dim: int,
        fuse_dim: int | None = None,
        num_heads: int | None = None,
    ):
        super().__init__()
        match fusion_type:
            case "concat":
                self.strategy = ConcatFusion(img_dim, secondary_dim)
            case "gated":
                if fuse_dim is None:
                    raise ValueError("fuse_dim required for gated fusion")
                self.strategy = GatedFusion(img_dim, secondary_dim, fuse_dim)
            case "cross_attention":
                if num_heads is None:
                    raise ValueError("num_heads required for cross_attention fusion")
                self.strategy = CrossAttentionFusion(img_dim, secondary_dim, num_heads)
            case _:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self, img_tokens: torch.Tensor, secondary_tokens: torch.Tensor
    ) -> torch.Tensor:
        return self.strategy(img_tokens, secondary_tokens)

    @property
    def requires_pooled(self) -> bool:
        return self.strategy.requires_pooled

    @property
    def output_dim(self) -> int:
        if isinstance(self.strategy, ConcatFusion):
            return self.strategy.img_dim + self.strategy.secondary_dim
        elif isinstance(self.strategy, GatedFusion):
            return self.strategy.fuse_dim
        elif isinstance(self.strategy, CrossAttentionFusion):
            return self.strategy.cross_attn.embed_dim
        raise ValueError(f"Unknown strategy: {type(self.strategy)}")
