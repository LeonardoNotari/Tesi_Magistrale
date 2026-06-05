import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerDecoderBlock(nn.Module):
    """
    Un singolo blocco: cross-attention su z + self-attention tra query.
    """
    def __init__(self, D, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(D, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, D),
        )

        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.norm3 = nn.LayerNorm(D)
        self.drop  = nn.Dropout(dropout)

    def forward(self, queries, memory):

        # cross-attention
        q2, _ = self.cross_attn(queries, memory, memory)
        queries = self.norm1(queries + self.drop(q2))

        # self-attention
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + self.drop(q2))

        # feed-forward
        queries = self.norm3(queries + self.drop(self.ff(queries)))
        return queries


class TransformerDecoder(nn.Module):

    def __init__(self, D=1024, num_queries=64, points_per_query=32,
                 num_blocks=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries      = num_queries
        self.points_per_query = points_per_query

        # proiezione di z in spazio D 
        self.z_proj = nn.Linear(D, D)

        # query learnable 
        self.queries = nn.Parameter(torch.randn(num_queries, D) * 0.02)

        # stack di blocchi transformer
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(D, num_heads, ff_dim=D*2, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # testa che genera i punti per ogni query
        self.point_head = nn.Sequential(
            nn.Linear(D, 256),
            nn.GELU(),
            nn.Linear(256, points_per_query * 3),
        )

    def forward(self, z):

        B = z.shape[0]

        # proietta z e aggiunge dimensione sequenza → (B, 1, D)
        memory = self.z_proj(z).unsqueeze(1)

        # espandi le query per il batch → (B, Q, D)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # passa attraverso i blocchi transformer
        for block in self.blocks:
            queries = block(queries, memory)

        # genera i punti: (B, Q, P*3) → (B, Q*P, 3)
        points = self.point_head(queries)
        points = points.reshape(B, self.num_queries * self.points_per_query, 3)
        return points

