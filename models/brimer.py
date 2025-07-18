import torch
from torch import nn
from dataset_briges import load_all
import numpy as np
import os

vocab, vocab_size, token2id, id2token, data_loader = load_all()  # 字典大小为30

def get_sinusoidal_positional_encoding(seq_len, d_model):
	pos_enc = torch.zeros(seq_len, d_model)
	position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

	pos_enc[:, 0::2] = torch.sin(position * div_term)
	pos_enc[:, 1::2] = torch.cos(position * div_term)
	return pos_enc  # shape: [seq_len, d_model]

class Brimer(nn.Module):
    def __init__(
        self,
        d_model=256,
        max_len=335,
        num_classes=30,
        num_layers_encoder=4,
        num_layers_decoder=4,
        num_heads=8,
        dim_feedforward=512,
        dropout=0.5,  # 不抑制
        read_embedding=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional Encoding
        self.positional_encoding = self._get_positional_encoding(max_len, d_model)
        n_embed = nn.Parameter(torch.randn(28, d_model))
        e_embed = nn.Parameter(torch.randn(28, d_model))
        s_embed = nn.Parameter(torch.randn(28, d_model))
        w_embed = nn.Parameter(torch.randn(28, d_model))
        jiao_embed = nn.Parameter(torch.randn(45, d_model))
        chu_embed = nn.Parameter(torch.randn(156, d_model))
        # 不更新的头部 3 个 + 尾部 18 个零向量，共 (21, d_model)
        self.register_buffer("head_zeros", torch.zeros(4, d_model))  # 不可训练
        self.register_buffer("zhuang_zeros", torch.zeros(18, d_model))
        self.bridge_embedding = torch.cat(
            [
                self.head_zeros,
                n_embed,
                e_embed,
                s_embed,
                w_embed,
                self.zhuang_zeros,
                jiao_embed,
                chu_embed,
            ],
            dim=0,
        ).unsqueeze(0)
        if read_embedding:
            self.load_all_embeddings()
        # print(self.bridge_embedding.size(),self.positional_encoding.size())

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用 [B, L, D] 顺序
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers_encoder
        )
        # 新增解码器部分，4层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers_decoder)

        # 输出分类头：可以是线性 + softmax（多分类）或线性 + sigmoid（二分类）
        self.classes = nn.Linear(d_model, num_classes) # 专门解码前面的三个预测

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
        x = self.embedding(x) # 首先获得嵌入表示
        # 加上位置编码和桥牌编码
        x = x + self.positional_encoding + self.bridge_embedding  # [B, 335, 256]
        # 编码器
        encoder_output = self.transformer_encoder(x)  # [B, 335, 256]
        # 解码器，输入 tgt 这里用 encoder_output 作为示例
        decoder_output = self.transformer_decoder(
            tgt=encoder_output,  # 你也可以传别的 tgt
            memory=encoder_output,
        )  # [B, 335, 256]
        # 取 decoder 输出指定位置的 token 表示
        if pred_position:
            x_cls = decoder_output[:, pred_position, :]  # [B, len(pred_position), 256]
        else:
            x_cls = decoder_output[:, 1:4, :]  # [B, 3, 256]
        # 分类
        out = self.classes(x_cls)  # [B, num_classes]
        return out


    """
    特殊字符<mask>为不知道的字段，训练时有一个目标预测。<None>为已经没有的字段，不需要预测。<cls>为分类任务的目标，这里有三个（角色，花色，大小）。
    一个BridgeItem应该可以导出52个训练数据（预测每一次出牌，另外在预测的同时增加预测其他人手牌的任务）。
    最终导出为List[list]数据，外层长度为52，内层长度应该为334。保存为pkl数据，方便之后读取。

    在后续的处理中加入预测不同任务时的提示词<pre>和<cls>。所以后续的所有索引全部+=1

    最终索引0-2是预测，3 4-29 30 是第n的手牌 31 32-57 58 是e的手牌
    59 60-85 86是s的手牌 87 88-113 114 是w的手牌

    115-116 117-132是庄家
    133-177 是叫牌

    178-180 1轮
    181-183 2轮
    .....
    331-333 13轮
    """


def extract_use_x_y(xs, labels):
    all_use_x = []
    all_use_y = []

    for x, label in zip(xs, labels):
        task_type = x[0].item()
        use_x_tensor = []
        use_y_tensor = []

        if task_type == 3:
            sub_label = x[134:179]
            indices = (sub_label == 1).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                use_x = indices + 134
                use_y = use_x - 1
                use_x_tensor.append(use_x)
                use_y_tensor.append(use_y)

        elif task_type == 4:
            indices = (x == 1).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                use_x = indices
                use_y = use_x - 1
                use_x_tensor.append(use_x)
                use_y_tensor.append(use_y)

        if use_x_tensor:  # 确保有内容
            use_x_tensor[0] = torch.cat(
                [torch.tensor([1, 2, 3], dtype=torch.long), use_x_tensor[0]]
            )
            use_y_tensor[0] = torch.cat([use_y_tensor[0][-3:].detach().clone(), use_y_tensor[0]])
            assert len(use_x_tensor) == len(use_y_tensor) and len(use_x_tensor[0]) == len(use_y_tensor[0])
            all_use_x.append(use_x_tensor[0])
            all_use_y.append(use_y_tensor[0])
    return all_use_x, all_use_y
if __name__ == "__main__":
    brimer = Brimer(read_embedding=True)
    for xs, labels in data_loader:
        use_x,use_y = extract_use_x_y(xs,labels) # 拼接成功
        # print(use_y,"\n",use_x)
        # print(labels,use_y)
        # print(brimer(x))
        break # 可以正常前向