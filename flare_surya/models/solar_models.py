import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, final_act=True):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.final_act = final_act

        # The Shortcut path (identity or projection)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Addition step
        out += residual
        if self.final_act:
            return self.act(out)
        return out


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, final_act=True):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.final_act = final_act

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
        if self.final_act:
            return self.act(out)
        return out


class SolarTokenizer1D(nn.Module):
    def __init__(self, in_channels=2, embed_dim=768):
        super().__init__()
        self.layer1 = ResidualBlock1D(in_channels, embed_dim, kernel_size=7)
        self.layer2 = ResidualBlock1D(embed_dim, embed_dim, kernel_size=5)

    def forward(self, x):
        # x: [Batch, 1, seq_len]
        x = self.layer1(x)
        x = self.layer2(x)

        # Prepare for Transformer [Batch, Seq_Len, Dim]
        return x.transpose(1, 2)


class SolarTokenizer2D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, image_size=224, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SolarViTBlock1D(nn.Module):
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
        # x shape: [Batch, seq_len, embed_dim]

        # 1. Attention path with Residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # 2. MLP path with Residual
        x = x + self.mlp(self.norm2(x))

        return x


class SolarSequenceEncoder(nn.Module):
    def __init__(self, seq_len=1440, embed_dim=768, depth=4, num_heads=12):
        super().__init__()

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Standard ViT practice is to drop after adding pos_embed
        self.pos_drop = nn.Dropout(p=0.1)

        # Stack the ViT blocks
        self.blocks = nn.ModuleList(
            [
                SolarViTBlock1D(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ]
        )

        # Final norm (standard in ViT before pooling or fusion)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for the positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x is the output from SecondaryTokenizer: [Batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class SolarEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        seq_len=1440,
        embed_dim=768,
        depth=4,
        num_heads=12,
        data_type="1d",
        image_size=224,
        patch_size=16,
    ):
        super().__init__()
        self.data_type = data_type

        if data_type == "1d":
            self.tokenizer = SolarTokenizer1D(in_channels, embed_dim)
            # For 1D, seq_len is the sequence length
            self.encoder = SolarSequenceEncoder(seq_len, embed_dim, depth, num_heads)
        elif data_type == "2d":
            self.tokenizer = SolarTokenizer2D(
                in_channels, embed_dim, image_size, patch_size
            )
            # For 2D, seq_len is the number of patches
            num_patches = (image_size // patch_size) ** 2
            self.encoder = SolarSequenceEncoder(
                num_patches, embed_dim, depth, num_heads
            )
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def forward(self, x):
        token = self.tokenizer(x)
        embedding = self.encoder(token)

        return embedding


class SolarSequenceDecoder(nn.Module):
    def __init__(self, seq_len=1440, embed_dim=768, depth=2, num_heads=12):
        super().__init__()

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Standard ViT practice is to drop after adding pos_embed
        self.pos_drop = nn.Dropout(p=0.1)

        # Stack the ViT blocks
        self.blocks = nn.ModuleList(
            [
                SolarViTBlock1D(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ]
        )

        # Final norm (standard in ViT before pooling or fusion)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for the positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x is the output from Encoder: [Batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class SolarDetokenizer1D(nn.Module):
    def __init__(self, in_channels=2, embed_dim=768):
        super().__init__()
        # Invert the tokenization: embed_dim -> in_channels
        # We use transposed convolutions (ConvTranspose1d) to upsample
        self.layer1 = ResidualBlock1D(embed_dim, embed_dim, kernel_size=5)
        self.layer2 = ResidualBlock1D(embed_dim, in_channels, kernel_size=7, final_act=False)

    def forward(self, x):
        # x: [Batch, seq_len, embed_dim] - Output from SequenceDecoder
        # We need to convert back to [Batch, in_channels, seq_len]

        # First convert to [Batch, embed_dim, seq_len]
        x = x.transpose(1, 2)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SolarDetokenizer2D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=768, image_size=512):
        super().__init__()
        self.image_size = image_size
        patch_size = 16
        self.num_patches_per_side = image_size // patch_size

        # 1. Bottleneck: Drastically reduce channels at low resolution (32x32)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # 2. Progressive Upsampling Stage 1 (32 -> 128)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # 3. Progressive Upsampling Stage 2 (128 -> 512)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # 4. Final Projection to image channels
        self.final_conv = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        B, num_patches, D = x.shape
        H = W = self.num_patches_per_side

        # [B, 1024, 768] -> [B, 768, 32, 32]
        x = x.transpose(1, 2).reshape(B, D, H, W)

        # Apply progressive decoding
        x = self.bottleneck(x)  # -> [B, 128, 32, 32]
        x = self.upsample1(x)   # -> [B, 64, 128, 128]
        x = self.upsample2(x)   # -> [B, 32, 512, 512]
        x = self.final_conv(x)  # -> [B, in_channels, 512, 512]

        return x


class SolarDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        seq_len=1440,
        embed_dim=768,
        depth=2,
        num_heads=12,
        data_type="1d",
        image_size=224,
    ):
        super().__init__()
        self.data_type = data_type

        self.sequence_decoder = SolarSequenceDecoder(
            seq_len, embed_dim, depth, num_heads
        )

        if data_type == "1d":
            self.detokenizer = SolarDetokenizer1D(in_channels, embed_dim)
        elif data_type == "2d":
            self.detokenizer = SolarDetokenizer2D(in_channels, embed_dim, image_size)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def forward(self, x):
        x = self.sequence_decoder(x)
        reconstruction = self.detokenizer(x)
        return reconstruction

    def decode_sequence(self, x):
        return self.sequence_decoder(x)
