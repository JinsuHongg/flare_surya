import torch
import torch.nn as nn


class ResidualFluxBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # The Shortcut path (identity or projection)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Addition step
        out += residual
        return self.act(out)


class FluxTokenizer(nn.Module):
    def __init__(self, in_channels=2, embed_dim=768):
        super().__init__()
        self.layer1 = ResidualFluxBlock(in_channels, embed_dim, kernel_size=7)
        self.layer2 = ResidualFluxBlock(embed_dim, embed_dim, kernel_size=5)

    def forward(self, x):
        # x: [Batch, 1, 1440]
        x = self.layer1(x)
        x = self.layer2(x)

        # Prepare for Transformer [Batch, Seq_Len, Dim]
        return x.transpose(1, 2)


class ViTBlock1D(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self Attention
        # batch_first=True ensures we use [Batch, Seq_Len, Dim]
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # The MLP (Feed-Forward) block with 4x expansion
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # Standard activation for ViT architectures
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: [Batch, 1440, 768]

        # 1. Attention path with Residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # 2. MLP path with Residual
        x = x + self.mlp(self.norm2(x))

        return x


class TimeseriesTransformerEncoder(nn.Module):
    def __init__(self, seq_len=1440, embed_dim=768, depth=4, num_heads=12):
        super().__init__()

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Standard ViT practice is to drop after adding pos_embed
        self.pos_drop = nn.Dropout(p=0.1)

        # Stack the ViT blocks
        self.blocks = nn.ModuleList(
            [ViTBlock1D(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
        )

        # Final norm (standard in ViT before pooling or fusion)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for the positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x is the output from FluxTokenizer: [Batch, 1440, 768]

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class FluxFormer(nn.Module):
    def __init__(
        self, in_channels=1, seq_len=1440, embed_dim=768, depth=4, num_heads=12
    ):
        super().__init__()
        self.tokenizer = FluxTokenizer(in_channels, embed_dim)
        self.encoder = TimeseriesTransformerEncoder(
            seq_len, embed_dim, depth, num_heads
        )

    def forward(self, x):
        token = self.tokenizer(x)
        embedding = self.encoder(token)

        return embedding
