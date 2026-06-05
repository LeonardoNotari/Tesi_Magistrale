import torch
import torch.nn as nn

class ULIPDecoder_v2(nn.Module):
    def __init__(self, emb_dim, num_points):
        super().__init__()
        self.num_points = num_points

        hidden = 2048

        self.fc1 = nn.Linear(emb_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)

        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)

        self.fc3 = nn.Linear(hidden, hidden)
        self.ln3 = nn.LayerNorm(hidden)

        self.fc4 = nn.Linear(hidden, hidden)
        self.ln4 = nn.LayerNorm(hidden)

        self.out = nn.Linear(hidden, num_points * 3)

        self.act = nn.GELU()

    def forward(self, z):
        # Layer 1
        x = self.act(self.ln1(self.fc1(z)))

        # Layer 2 + skip
        x2 = self.act(self.ln2(self.fc2(x)))
        x = x + x2

        # Layer 3 + skip
        x3 = self.act(self.ln3(self.fc3(x)))
        x = x + x3

        # Layer 4 + skip
        x4 = self.act(self.ln4(self.fc4(x)))
        x = x + x4

        # Output
        out = self.out(x)
        out = out.view(-1, self.num_points, 3)
        return out
