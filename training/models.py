
from __future__ import annotations

import torch
from torch import nn

MODEL_SPECS = {
    "A": {"channels": (8, 16, 24), "hidden": 128, "parameters": 83888},
    "B": {"channels": (6, 12, 16), "hidden": 64, "parameters": 29812},
    "C": {"channels": (8, 16, 32), "hidden": 96, "parameters": 83736},
}

class TinyCaptchaCNN(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        name = model_name.upper()
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model_name!r}; expected A, B or C")
        spec = MODEL_SPECS[name]
        c1, c2, c3 = spec["channels"]
        self.model_name = name
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=True)
        self.fc1 = nn.Linear(c3 * 3 * 8, spec["hidden"], bias=True)
        self.fc2 = nn.Linear(spec["hidden"], 40, bias=True)
        self.activation = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.activation(self.conv1(images))
        values = self.activation(self.conv2(values))
        values = self.activation(self.conv3(values))
        values = torch.flatten(values, start_dim=1)
        values = self.activation(self.fc1(values))
        return self.fc2(values).reshape(-1, 4, 10)

def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
