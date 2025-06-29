import torch
import torch.nn as nn


class FaceClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(FaceClassifierNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)
