import torch
from torch import nn
from dataset_briges import load_all
vocab, vocab_size, token2id, id2token, data_loader = load_all() # 字典大小为30

class Brimer(nn.Module):
	def __init__(
		self,
		d_model=256,
		max_len=335,
		num_classes=30,
		num_layers=4,
		num_heads=8,
		dim_feedforward=512,
		dropout=0.1,
	):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, d_model)

		# Positional Encoding
		self.positional_encoding = self._get_positional_encoding(max_len, d_model)

		# Transformer Encoder Layer
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=num_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,  # 使用 [B, L, D] 顺序
		)
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer, num_layers=num_layers
		)

		# 输出分类头：可以是线性 + softmax（多分类）或线性 + sigmoid（二分类）
		self.classifier = nn.Linear(d_model, num_classes)

	def _get_positional_encoding(self, seq_len, d_model):
		# 简单的可学习位置编码（可改为正弦位置编码）
		return nn.Parameter(torch.randn(1, seq_len, d_model))

	def _get_roles_encoding(self, seq_len, d_model):
		# 简单的可学习角色编码，添加到n,e,s,w四个角色手牌位置
		return nn.Parameter(torch.randn(1, seq_len, d_model))

	def _get_jiao_encoding(self, seq_len, d_model):
		# 简单的可学习叫牌编码，添加到叫牌位置
		return nn.Parameter(torch.randn(1, seq_len, d_model))

	def _get_chu_encoding(self, seq_len, d_model):
		# 简单的可学习出牌编码，添加到出牌位置
		return nn.Parameter(torch.randn(1, seq_len, d_model))

	@staticmethod
	def tokens_to_ids(tokens, token2id):
		return [token2id.get(tok) for tok in tokens]

	@staticmethod
	def ids_to_token(tokens, id2token):
		return [id2token.get(tok) for tok in tokens]

	def forward(self, x):  # x: [B, 335, 256]
		x = x + self.positional_encoding  # 加上位置编码
		x = self.transformer_encoder(x)   # 输出仍是 [B, 335, 256]

		# 常见做法1：取第一个 token 输出做分类（类似 BERT）
		x_cls = x[:, 0, :]  # [B, 256]

		# 常见做法2：全局平均池化（根据任务选择）
		# x_cls = x.mean(dim=1)

		out = self.classifier(x_cls)  # [B, num_classes]
		return out

