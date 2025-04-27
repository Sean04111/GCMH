import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ImgNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        # 新增残差网络
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_encode = nn.Linear(4096, code_len)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.attention = nn.Sequential(
            nn.Linear(4096, 256),
            nn.Tanh(),
            nn.Linear(256, 4096),
            nn.Sigmoid()
        )

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
       # torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.1)  

    def forward(self, x):

        x = x.view(x.size(0), -1)

        identity = x
        feat1 = self.relu(self.fc1(x))
        feat2 = self.relu(self.fc2(feat1) + identity)
        #feat1 = feat1 + self.relu(self.fc2(self.dropout(feat1)))
        hid = self.fc_encode(self.dropout(feat2))
        code = torch.tanh(self.alpha * hid)

        return (x, x), code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
