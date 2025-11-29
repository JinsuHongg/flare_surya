from torch import nn
from torchvision.ops import MLP


class SuryaHead(nn.Module):
    def __init__(
            self,
            in_feature,
            layer_type, 
            layer_dict
            ):
        super().__init__()

        # Registry for YAML → PyTorch class mapping
        NORM_LAYERS = {
            "LayerNorm": nn.LayerNorm,
            "BatchNorm1d": nn.BatchNorm1d,
            None: None,
            "None": None,
        }

        ACTIVATIONS = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            None: None,
            "None": None,
        }

        match layer_type:
            case "mlp":
                self.head = MLP(
                    in_channels=in_feature,
                    hidden_channels=layer_dict["hidden_channels"],
                    norm_layer=NORM_LAYERS[layer_dict.get("norm_layer", None)],
                    activation_layer=ACTIVATIONS[layer_dict.get("activation_layer", None)],
                    bias=layer_dict.get("bias", True),
                    dropout=layer_dict.get("dropout", 0.0)
                )
            case _:
                raise ValueError(f"Unknown layer type: {layer_type}")

    def forward(self, batch):
        return self.head(batch)
