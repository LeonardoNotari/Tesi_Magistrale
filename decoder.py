import torch.nn as nn

class ULIPDecoder(nn.Module):
    def __init__(self, emb_dim, num_points):
        super().__init__()
        self.num_points = num_points

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 3)
        )

    def forward(self, z):
        out = self.mlp(z)               # (B, num_points*3)
        out = out.view(-1, self.num_points, 3)
        return out
