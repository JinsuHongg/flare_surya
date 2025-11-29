import torch
from torch import nn
import torch.nn.functional as F
import lightning as L


class BaseModule(L.LightningModule):
    def __init__(self, optimizer_dict):
        super().__init__()
        self.optimizer_dict = optimizer_dict

    def configure_optimizers(self):
        opt_type = self.optimizer_dict.get("type", "adam")
        lr = self.optimizer_dict.get("lr", 1e-3)
        weight_decay = self.optimizer_dict.get("weight_decay", 0.0)

        match opt_type:
            case "adam":
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case _:
                raise ValueError(f"Unknown optimizer type: {opt_type}")

        return optimizer
