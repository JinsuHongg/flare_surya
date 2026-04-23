import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet18, self).__init__()

        # 1. Load the pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        merged_channels = in_channels * time_steps

        # 2. Extract the original pretrained weights before overwriting the layer
        # Shape is [64, 3, 7, 7] (out_channels, in_channels, kernel_size, kernel_size)
        pretrained_weights = self.resnet.conv1.weight.clone()

        # 3. Create the new conv layer
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 4. Smart Initialization: Average the RGB weights and repeat them
        with torch.no_grad():
            # Average across the 3 RGB channels to get a generic "edge detector" [64, 1, 7, 7]
            avg_weight = pretrained_weights.mean(dim=1, keepdim=True)

            # Repeat this generic filter across your new number of channels
            # We scale it by (3 / merged_channels) so the overall output magnitude remains stable
            repeated_weights = avg_weight.repeat(1, merged_channels, 1, 1) * (
                3.0 / merged_channels
            )

            # Inject it into your new layer
            self.resnet.conv1.weight.copy_(repeated_weights)

        # 5. Remove the final classification layer (keep everything else the same)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # 6. Add custom classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        x = x["ts"]
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 512, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 512

        # Classification
        features = self.dropout(features)
        output = self.classifier(features)
        # output = torch.sigmoid(output)

        return output.squeeze(-1)
