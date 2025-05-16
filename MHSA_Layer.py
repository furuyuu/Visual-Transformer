import torch
import torch.nn as nn
import torch.nn.functional as F

from Input_Layer import VitInputLayer

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim:int=384, # 埋め込み後のベクトルの長さ
                 head:int=3,      # ヘッドの数
                 dropout:float=0. # ドロップアウト率
    ):
        
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        # q, k, vに埋め込むための線形層
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # ドロップアウト層
        self.atten_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, _ =z.size()

        # 埋め込み
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q, k, vをヘッドの個数に分ける
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # Self-Attentionができるように
        # (B, N, h, D//h) -> (B, h, N, D//h)に変更
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # kを転置
        k_T = k.transpose(2, 3)
        # 内積
        dots = (q @ k_T) / self.sqrt_dh

        atten = F.softmax(dots, dim=-1)

        atten = self.atten_drop(atten)

        # 加重和
        out = atten @ v

        out = out.transpose(1, 2)

        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層
        out = self.w_o(out)
        return out
    

