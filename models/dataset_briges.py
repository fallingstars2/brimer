import os
import pickle
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset 类
class Brimer_Dataset(Dataset):
    def __init__(self, data):
        self.data = data  # data 是 [(x1, label1), (x2, label2), ...]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, label = self.data[idx]
        return x, label

def get_datas(file_name="bridges_dataset"):
    # 获取当前脚本所在的绝对路径
    current_dir = os.path.abspath(os.path.dirname(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    parent_dir += f"/bridge/{file_name}"
    with open(parent_dir, 'rb') as f:
        data = pickle.load(f)
    print(f"一共加载了 {len(data)} 条数据")
    return data

# 构造 Dataset 和 DataLoader（需你补全）
def get_dataloader(data, batch_size=16, shuffle=True):
    dataset = Brimer_Dataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True  # 舍弃最后不足一个batch的数据
    )
    return dataloader

vocab = ["A","1", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K","S","H","D","C","n", "e", "s", "w", "X", "P", "N","<cls>", "<mask>", "<None>", "<jiao>", "<pre>"]

token2id = {tok: i for i, tok in enumerate(vocab)}
id2token = {i: tok for tok, i in token2id.items()}

vocab_size = len(vocab)

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

if __name__ == '__main__':
    load_all(file_name="bridges_dataset.pkl")