import torch
import torch.nn as nn

from MHSA_Layer import MultiHeadSelfAttention

class VitEncoderBlock(nn.Module):
    def __init__(
            self,
            emb_dim:int=384,
            head:int=8,
            hidden_dim:int=384*4,
            dropout:float=0.
    ):
        
        super(VitEncoderBlock, self).__init__()

        self.ln1 = nn.LayerNorm(emb_dim)

        self.msa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.ln1(z)) + z
        out = self.mlp(self.ln2(out)) + out

        return out




