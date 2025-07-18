import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

vocab = [
    "<cls>",
    "<mask>",
    "<None>",
    "<jiao>",
    "<pre>",
    "A",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "J",
    "Q",
    "K",
    "S",
    "H",
    "D",
    "C",
    "n",
    "e",
    "s",
    "w",
    "X",
    "P",
    "N",
]

token2id = {tok: i for i, tok in enumerate(vocab)}
id2token = {i: tok for tok, i in token2id.items()}

vocab_size = len(vocab) # 30


# 自定义 Dataset 类
class Brimer_Dataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab  # 你需要一个词表把token转成数字id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_tokens, y_tokens = self.data[idx]
        # print(x_tokens)
        # 把token转id，例如：
        x_ids = [self.vocab.get(token) for token in x_tokens]
        y_ids = [self.vocab.get(token) for token in y_tokens]

        assert all(id_ is not None for id_ in x_ids), "x_ids contains None!"
        assert all(id_ is not None for id_ in y_ids), "y_ids contains None!"
        # 转成Tensor
        x_tensor = torch.tensor(x_ids, dtype=torch.long).to("cuda")
        y_tensor = torch.tensor(y_ids, dtype=torch.long).to("cuda")

        return x_tensor, y_tensor


def get_datas(file_name="bridges_dataset.pkl"):
    # 获取当前脚本所在的绝对路径
    current_dir = os.path.abspath(os.path.dirname(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    parent_dir += f"/bridge/{file_name}"
    with open(parent_dir, "rb") as f:
        data = pickle.load(f)
    print(f"一共加载了 {len(data)} 条数据")
    return data


# 构造 Dataset 和 DataLoader（需你补全）
def get_dataloader(data, batch_size=4, shuffle=True):
    # print(data[0])
    # for d,d2 in data:
    #     print(d)
    #     break
    dataset = Brimer_Dataset(data, token2id)
    # print(dataset[0:1])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # 舍弃最后不足一个batch的数据
    )
    return dataloader


def load_all(file_name="bridges_dataset.pkl"):
    datas = get_datas(file_name)
    # print(token2id)
    # for data in datas: # 经过测试能涵盖所有的嵌入
    #     r = []
    #     for i in range(len(data[0])):
    #         d = data[0][i]
    #         if token2id.get(d,None) == None:
    #             print(f"{i}处有漏掉的词元{d}")
    #         r.append(token2id.get(d))
    #     # print(r)
    return vocab, vocab_size, token2id, id2token, get_dataloader(datas)


if __name__ == "__main__":
    da = get_datas()
    # print(da[:1])
    # for x,y in da:
    #     print(x,y)
    #     break
    data_loader = get_dataloader(da)
    # for d in data_loader:
    #     print(d)
    #     break
