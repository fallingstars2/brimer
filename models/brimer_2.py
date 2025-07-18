import torch
from torch import nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, device="cuda"):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.device = device

        self.depth = d_model // nhead

        # Linear transformations for query, key, and value
        self.query = nn.Linear(d_model, d_model).to(device)
        self.key = nn.Linear(d_model, d_model).to(device)
        self.value = nn.Linear(d_model, d_model).to(device)
        self.fc_out = nn.Linear(d_model, d_model).to(device)

        # Dropout and layer normalization
        self.dropout_layer = nn.Dropout(dropout).to(device)
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, query, key, value, mask=None):
        query, key, value = (
            query.to(self.device),
            key.to(self.device),
            value.to(self.device),
        )

        batch_size = query.size(0)

        # Linear projections
        Q = (
            self.query(query)
            .view(batch_size, -1, self.nhead, self.depth)
            .transpose(1, 2)
        )  # [B, nhead, L, depth]
        K = (
            self.key(key).view(batch_size, -1, self.nhead, self.depth).transpose(1, 2)
        )  # [B, nhead, L, depth]
        V = (
            self.value(value)
            .view(batch_size, -1, self.nhead, self.depth)
            .transpose(1, 2)
        )  # [B, nhead, L, depth]

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.depth
        )  # [B, nhead, L, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Output
        out = torch.matmul(attention_weights, V)  # [B, nhead, L, depth]
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # [B, L, d_model]

        # Final linear layer
        out = self.fc_out(out)
        out = self.layer_norm(out + query)  # Residual connection
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, device="cuda"):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.fc2 = nn.Linear(dim_feedforward, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.layer_norm = nn.LayerNorm(d_model).to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.layer_norm(out + x)  # Residual connection
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, device="cuda"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, nhead, dropout, device)
        self.feedforward = FeedForward(d_model, dim_feedforward, dropout, device)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        # Self Attention (Encoder side)
        src = self.self_attention(src, src, src, mask=src_mask)

        # Feed Forward layer
        src = self.feedforward(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, device="cuda"
    ):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, device
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)
        return output


class CustomTransformerEncoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, device="cuda"
    ):
        super(CustomTransformerEncoder, self).__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, num_layers, dropout, device
        )

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        return self.encoder(src, src_mask)


def get_sinusoidal_positional_encoding(seq_len, d_model, device="cuda"):
    pos_enc = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-np.log(10000.0) / d_model)
    )

    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc  # shape: [seq_len, d_model]



class Brimer2(nn.Module):
    def __init__(
        self,
        d_model=256,
        max_len=335,
        num_classes=30,
        num_layers_encoder=6,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.02,  # 不抑制
        read_embedding=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(30, d_model).to("cuda")
        # Positional Encoding
        self.positional_encoding = self._get_positional_encoding(max_len, d_model).to("cuda")
        self.n_embed = nn.Parameter(torch.randn(28, d_model).to("cuda").detach())
        self.e_embed = nn.Parameter(torch.randn(28, d_model).to("cuda").detach())
        self.s_embed = nn.Parameter(torch.randn(28, d_model).to("cuda").detach())
        self.w_embed = nn.Parameter(torch.randn(28, d_model).to("cuda").detach())
        self.jiao_embed = nn.Parameter(torch.randn(45, d_model).to("cuda").detach())
        self.chu_embed = nn.Parameter(torch.randn(156, d_model).to("cuda").detach())
        # 不更新的头部 3 个 + 尾部 18 个零向量，共 (21, d_model)
        self.register_buffer("head_zeros", torch.zeros(4, d_model).to("cuda").detach())  # 不可训练
        self.register_buffer("zhuang_zeros", torch.zeros(18, d_model).to("cuda").detach())
        self.bridge_embedding = torch.cat(
            [
                self.head_zeros,
                self.n_embed,
                self.e_embed,
                self.s_embed,
                self.w_embed,
                self.zhuang_zeros,
                self.jiao_embed,
                self.chu_embed,
            ],
            dim=0,
        ).unsqueeze(0).detach()
        if read_embedding:
            self.load_all_embeddings()
        # print(self.bridge_embedding.size(),self.positional_encoding.size())
        self.bridge_embedding = self.bridge_embedding.to("cuda")
        # Transformer Encoder Layer
        self.transformer_encoder=CustomTransformerEncoder(d_model, num_heads, dim_feedforward, num_layers_encoder, dropout)
        # 输出分类头：可以是线性 + softmax（多分类）或线性 + sigmoid（二分类）
        self.classes = nn.Linear(d_model, num_classes).to("cuda")  # 专门解码前面的三个预测

    def _get_positional_encoding(self, seq_len, d_model):
        # 简单的可学习位置编码
        pe_init = get_sinusoidal_positional_encoding(seq_len, d_model)
        return nn.Parameter(pe_init.unsqueeze(0))

    def save_all_embeddings(self, path="all_embeddings.pt"):
        # 把所有需要保存的权重打包成tuple或dict
        all_embedding = (
            self.embedding.weight.data.cpu(),  # nn.Embedding的权重
            self.positional_encoding.data.cpu(),
            self.bridge_embedding.data.cpu(),
        )
        torch.save(all_embedding, path)
        print(f"权重保存到 {path}")

    def load_all_embeddings(self, path="all_embeddings.pt"):
        all_embedding = torch.load(path)
        self.embedding.weight.data.copy_(all_embedding[0])
        self.positional_encoding.data.copy_(all_embedding[1])
        self.bridge_embedding.data.copy_(all_embedding[2])
        print(f"权重从 {path} 加载完成")

    @staticmethod
    def tokens_to_ids(tokens, token2id):
        return [token2id.get(tok) for tok in tokens]

    @staticmethod
    def ids_to_token(tokens, id2token):
        return [id2token.get(tok) for tok in tokens]

    def forward(self, x, pred_position=None):  # x: [B, 335, 256]
        x = self.embedding(x)  # 首先获得嵌入表示
        # 加上位置编码和桥牌编码
        x = x + self.positional_encoding.detach() + self.bridge_embedding.detach()  # [B, 335, 256]
        # 编码器
        encoder_output = self.transformer_encoder(x)  # [B, 335, 256]
        # # 分类
        out = self.classes(encoder_output)  # [B,335,num_classes]
        return out
