from torch.utils.data import Dataset
import numpy as np


class BatchData(DataSet):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels
        
    def __getitem__(self, index):
        image = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return image, text, label, index

    def __len__(self):
        length = len(self.images)
        # 强制要求images和labels的长度相等
        assert len(self.images) ==  len(self.labels)
        return length