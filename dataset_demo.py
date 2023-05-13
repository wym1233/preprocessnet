import torch
from PIL import Image
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_dir = data_path
        with open(data_path, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line=self.lines[idx]
        path, num = line.split(' ')
        num = float(num)
        return path, num
path='D:/bitahubdownload/bpp_25_test.txt'
dataset=BaseDataset(data_path=path)
print(dataset.__getitem__(0))
print(len(dataset))
