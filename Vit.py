import torch
import torch.nn as nn

from Input_Layer import VitInputLayer
from MHSA_Layer import MultiHeadSelfAttention
from Encoder_Brock import VitEncoderBlock

class Vit(nn.Module):
    def __init__(self,
                 in_channels:int=1,
                 num_classes:int=10,    # 画像分類のクラス数
                 emb_dim:int=384,       # 埋め込み後のベクトルの長さ
                 num_patch_row:int=2,   # 一辺のパッチの数
                 image_size:int=28,     # 入力画像の一辺の大きさ（高さと幅は同じとする）
                 num_blocks:int=7,
                 head:int=8,
                 hidden_dim:int=384*4,  # Encoder BlockのMLPにおける中間層のベクトルの長さ
                 dropout:float=0.
        ):

        super(Vit, self).__init__()

        # Input Layer
        self.input_layer = VitInputLayer(
            in_channels,
            emb_dim,
            num_patch_row,
            image_size
        )

        # Encoder Block（多段）
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数：
            x：ViTへの入力画像 形状は（B, C, H, W）
                B：バッチサイズ

        返り値：
            out：ViTの出力 形状は（B, M）
                M：クラス数
        """

        out = self.input_layer(x)
        out = self.encoder(out)
        cls_token = out[:, 0]
        pred = self.mlp_head(cls_token)
        return pred