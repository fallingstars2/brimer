import torch
from torch import nn
from dataset_briges import load_all
import numpy as np
import os

vocab, vocab_size, token2id, id2token, data_loader = load_all()  # 字典大小为30
batch_size = 4


def get_sinusoidal_positional_encoding(seq_len, d_model):
    pos_enc = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )

    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc  # shape: [seq_len, d_model]


# class Brimer(nn.Module):
#     def __init__(
#         self,
#         d_model=256,
#         max_len=335,
#         num_classes=30,
#         num_layers_encoder=6,
#         num_layers_decoder=1,
#         num_heads=4,
#         dim_feedforward=512,
#         dropout=0.02,  # 不抑制
#         read_embedding=False,
#     ):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model).to("cuda")
#         # Positional Encoding
#         self.positional_encoding = self._get_positional_encoding(max_len, d_model).to("cuda")
#         self.n_embed = nn.Parameter(torch.randn(28, d_model).to("cuda"))
#         self.e_embed = nn.Parameter(torch.randn(28, d_model).to("cuda"))
#         self.s_embed = nn.Parameter(torch.randn(28, d_model).to("cuda"))
#         self.w_embed = nn.Parameter(torch.randn(28, d_model).to("cuda"))
#         self.jiao_embed = nn.Parameter(torch.randn(45, d_model).to("cuda"))
#         self.chu_embed = nn.Parameter(torch.randn(156, d_model).to("cuda"))
#         # 不更新的头部 3 个 + 尾部 18 个零向量，共 (21, d_model)
#         self.register_buffer("head_zeros", torch.zeros(4, d_model).to("cuda").detach())  # 不可训练
#         self.register_buffer("zhuang_zeros", torch.zeros(18, d_model).to("cuda").detach())
#         self.bridge_embedding = torch.cat(
#             [
#                 self.head_zeros,
#                 self.n_embed,
#                 self.e_embed,
#                 self.s_embed,
#                 self.w_embed,
#                 self.zhuang_zeros,
#                 self.jiao_embed,
#                 self.chu_embed,
#             ],
#             dim=0,
#         ).unsqueeze(0)
#         if read_embedding:
#             self.load_all_embeddings()
#         # print(self.bridge_embedding.size(),self.positional_encoding.size())
#         self.bridge_embedding = self.bridge_embedding.to("cuda")
#         # Transformer Encoder Layer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=num_heads,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,  # 使用 [B, L, D] 顺序
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers_encoder
#         )
#         # 新增解码器部分，4层
#         # decoder_layer = nn.TransformerDecoderLayer(
#         #     d_model=d_model,
#         #     nhead=num_heads,
#         #     dim_feedforward=dim_feedforward,
#         #     dropout=dropout,
#         #     batch_first=True,
#         # )
#         # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers_decoder)
#
#         # 输出分类头：可以是线性 + softmax（多分类）或线性 + sigmoid（二分类）
#         self.classes = nn.Linear(d_model, num_classes)  # 专门解码前面的三个预测
#
#     def _get_positional_encoding(self, seq_len, d_model):
#         # 简单的可学习位置编码
#         pe_init = get_sinusoidal_positional_encoding(seq_len, d_model)
#         return nn.Parameter(pe_init.unsqueeze(0))
#
#     def save_all_embeddings(self, path="all_embeddings.pt"):
#         # 把所有需要保存的权重打包成tuple或dict
#         all_embedding = (
#             self.embedding.weight.data.cpu(),  # nn.Embedding的权重
#             self.positional_encoding.data.cpu(),
#             self.bridge_embedding.data.cpu(),
#         )
#         torch.save(all_embedding, path)
#         print(f"权重保存到 {path}")
#
#     def load_all_embeddings(self, path="all_embeddings.pt"):
#         all_embedding = torch.load(path)
#         self.embedding.weight.data.copy_(all_embedding[0])
#         self.positional_encoding.data.copy_(all_embedding[1])
#         self.bridge_embedding.data.copy_(all_embedding[2])
#         print(f"权重从 {path} 加载完成")
#
#     @staticmethod
#     def tokens_to_ids(tokens, token2id):
#         return [token2id.get(tok) for tok in tokens]
#
#     @staticmethod
#     def ids_to_token(tokens, id2token):
#         return [id2token.get(tok) for tok in tokens]
#
#     def forward(self, x, pred_position=None):  # x: [B, 335, 256]
#         x = self.embedding(x)  # 首先获得嵌入表示
#         # 加上位置编码和桥牌编码
#         x = x + self.positional_encoding + self.bridge_embedding  # [B, 335, 256]
#         # 编码器
#         encoder_output = self.transformer_encoder(x)  # [B, 335, 256]
#         # # 分类
#         out = self.classes(encoder_output)  # [B,335,num_classes]
#         return out
#
#     """
#     特殊字符<mask>为不知道的字段，训练时有一个目标预测。<None>为已经没有的字段，不需要预测。<cls>为分类任务的目标，这里有三个（角色，花色，大小）。
#     一个BridgeItem应该可以导出52个训练数据（预测每一次出牌，另外在预测的同时增加预测其他人手牌的任务）。
#     最终导出为List[list]数据，外层长度为52，内层长度应该为334。保存为pkl数据，方便之后读取。
#
#     在后续的处理中加入预测不同任务时的提示词<pre>和<cls>。所以后续的所有索引全部+=1
#
#     最终索引0-2是预测，3 4-29 30 是第n的手牌 31 32-57 58 是e的手牌
#     59 60-85 86是s的手牌 87 88-113 114 是w的手牌
#
#     115-116 117-132是庄家
#     133-177 是叫牌
#
#     178-180 1轮
#     181-183 2轮
#     .....
#     331-333 13轮
#     """


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
                [torch.tensor([1, 2, 3], dtype=torch.long).to("cuda"), use_x_tensor[0]]
            )
            use_y_tensor[0] = torch.cat(
                [use_y_tensor[0][-3:].detach().clone().to("cuda"), use_y_tensor[0]]
            )
            assert len(use_x_tensor) == len(use_y_tensor) and len(
                use_x_tensor[0]
            ) == len(use_y_tensor[0])
            all_use_x.append(use_x_tensor[0])
            all_use_y.append(use_y_tensor[0])
    return all_use_x, all_use_y


import torch.nn.functional as F


def brimer_loss(pred_logits, targets, alpha=1.0, beta=0.3):
    """
    pred_logits: Tensor of shape [N, C] — logits for N samples, C classes
    targets:     Tensor of shape [N] — int64 class indices
    alpha:       Weight for first 3 samples
    beta:        Weight for remaining samples
    """
    # 交叉熵损失（不取平均）
    losses = F.cross_entropy(pred_logits, targets, reduction="none")  # [N]

    weights = torch.ones_like(losses) * beta
    weights[:3] = alpha  # 前三个样本赋值为 alpha

    weighted_loss = (losses * weights).mean()
    return weighted_loss


# def get_custom_optimizer(model: Brimer, base_lr: float):
#     # 位置编码（虽然是 register_buffer，若为 nn.Parameter 就能学习）
#     # 但这里你真正希望加大学习率的是 bridge_embedding
#     groups = []
#     # 位置嵌入（bridge_embedding 是由 Parameter 构建的）
#     groups.append(
#         {
#             "params": [
#                 param for name, param in model.named_parameters() if "embed" in name
#             ],
#             "lr": base_lr * 1.5,
#         }
#     )
#
#     # 分类头
#     groups.append({"params": model.classes.parameters(), "lr": base_lr * 2.0})
#
#     # Transformer 编码器层，从前到后学习率递减
#     num_layers = len(model.transformer_encoder.layers)
#     for i, layer in enumerate(model.transformer_encoder.layers):
#         # 比如 i = 0 ~ 5，靠前层学习率高
#         lr_scale = 1.5 - (i / (num_layers - 1))  # 从 1.5 线性降到 0.5
#         groups.append({"params": layer.parameters(), "lr": base_lr / lr_scale})
#
#     # 其他残余部分（如 embedding 层）
#     already_in = set()
#     for g in groups:
#         for p in g["params"]:
#             already_in.add(id(p))
#     rest = [p for p in model.parameters() if id(p) not in already_in]
#     if rest:
#         groups.append({"params": rest, "lr": base_lr})
#
#     return torch.optim.Adam(groups)


MODEL_PATH = "brimer_model.pt"
import matplotlib.pyplot as plt
from IPython.display import clear_output


def train_brimer(lr, epochs, brimer):
    optimizer = torch.optim.Adam(brimer.parameters(), lr=lr)
    all_losses = []

    total_steps = len(data_loader) * epochs
    warmup_steps = int(total_steps * 0.1)  # 前10%为warmup
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    step = 0

    plt.ion()  # 开启交互模式
    for epoch in range(epochs):
        epoch_losses = []
        for xs, labels in data_loader:
            use_x, use_y = extract_use_x_y(xs, labels)
            result = brimer(xs)
            loss = 0
            for r, u, l, y in zip(result, use_x, labels, use_y):
                loss += brimer_loss(r[u], l[y])
            loss = loss / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 👈 每一步更新学习率
            step += 1

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        all_losses.append(avg_loss)
        # ✅ 可视化：每个 epoch 更新一次图像
        clear_output(wait=True)
        plt.clf()
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Avg Loss")
        plt.plot(all_losses, label="avg loss")
        plt.legend()
        plt.pause(0.01)

        print(
            f"[Epoch {epoch + 1}] Avg Loss: {sum(epoch_losses) / len(epoch_losses):.4f}"
        )

    # 训练完毕保存模型
    torch.save(brimer.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    # 保存训练曲线图像
    plt.savefig("training_loss.png")

    plt.ioff()
    plt.show()


from torch.optim.lr_scheduler import LambdaLR
import math


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine decay

    return LambdaLR(optimizer, lr_lambda)


def show_brimer():
    from torchinfo import summary

    brimer = Brimer(read_embedding=False)
    brimer.eval()
    brimer = brimer.cuda()

    # 从数据加载器获取一个样本 batch
    for xs, labels in data_loader:
        xs = xs.cuda()  # 保证数据也在 GPU 上
        break  # 只取一组数据用于 summary

    # 展示模型结构
    model_summary = summary(
        brimer,
        input_data=xs,
        col_names=["input_size", "output_size", "num_params", "params_percent"],
        depth=4,  # 控制显示层级
        verbose=1,
    )

    # 保存到文件
    with open("brimer_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model_summary))


if __name__ == "__main__":
    # show_brimer()
    
    from brimer_2 import Brimer2
    
    brimer = Brimer2(read_embedding=False)
    # 尝试加载已有模型
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        brimer.load_state_dict(torch.load(MODEL_PATH))
    brimer.train()
    brimer = brimer.cuda()
    lr = 1e-3
    epochs = 2
    train_brimer(lr=lr, epochs=epochs, brimer=brimer)

    # for xs, labels in data_loader:
    #     use_x,use_y = extract_use_x_y(xs,labels) # 获得最终拿来做损失函数的索引位置
    #     # print(brimer(x))
    #     result = brimer(xs)
    #     loss = 0
    #     for r,u,l,y in zip(result,use_x,labels,use_y):
    #         # print(r[u],"\n\n",l[y])
    #         loss += brimer_loss(r[u], l[y])
    #         # print(loss)
    #     loss = loss / batch_size
    #     print(f"loss: {loss}")
    #     break # 可以正常前向
    # torch.save(brimer.state_dict(), MODEL_PATH)