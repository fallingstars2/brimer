# import torch
# from torch.onnx import dynamo_export
# from brimer import Brimer
# # 假设你已经定义了 Brimer 类，并且 vocab_size 已定义
# vocab_size = 30  # 你实际的词表大小
# brimer = Brimer(read_embedding=False)  # 创建模型实例
# brimer = brimer.cuda()  # 如果有 GPU
# # 加载模型权重
# model_path = "brimer_model.pt"
# state_dict = torch.load(model_path, map_location="cuda")
# brimer.load_state_dict(state_dict)
# brimer.eval()  # 切换为评估模式
# # 1. 创建模型并转到 eval 模式
# model = brimer.eval()
# # 2. 准备输入样例
# sample_input = torch.randn(1,355, 256)
#
# # 3. 使用 dynamo_export 导出 ONNX 模型（会返回一个 ExportOutput 对象）
# # noinspection PyDeprecation
# exported = dynamo_export(model, sample_input)
#
# # 4. 保存 ONNX 模型
# exported.save("my_model.onnx")



import torch
from brimer_2 import Brimer2
# 假设你已经定义了 Brimer 类，并且 vocab_size 已定义
vocab_size = 30  # 你实际的词表大小

brimer = Brimer2(read_embedding=False)  # 创建模型实例
brimer = brimer.cuda()  # 如果有 GPU

# 加载模型权重
model_path = "brimer_model.pt"
state_dict = torch.load(model_path, map_location="cuda")
brimer.load_state_dict(state_dict)
brimer.eval()  # 切换为评估模式

for name, param in brimer.named_parameters():
    param.requires_grad = False
for name, param in brimer.named_parameters(recurse=True):
    param.requires_grad = False

# 假设模型输入是 [batch_size, sequence_length]
batch_size = 1
sequence_length = 335  # 你的模型最大长度

dummy_input = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length), dtype=torch.long).cuda()

onnx_path = "brimer_model.onnx"

torch.onnx.export(
    brimer,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=14,  # 推荐用14或更高
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},  # 支持动态batch和序列长度
        "output": {0: "batch_size", 1: "sequence_length"},
    },
)

print(f"ONNX model saved to {onnx_path}")