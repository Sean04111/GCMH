# 预先处理数据
# 1. 特征提取
# 2. 数据清洗

import numpy as np
import torchvision.models as models


class FeatureExtractor:
    def __init__(self, model_name='resnet'):
        if model_name == 'resnet':
            self.model = models.resnet50(pretrained=True)
            self.model.eval()