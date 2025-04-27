# 预先处理数据
# 1. 特征提取
# 2. 数据清洗

import numpy as np
import torchvision.models as models
from config.config import Config
from utils.logger import log
from scipy.io import loadmat
import h5py
from keras.applications.vgg19 import VGG19
from keras.models import Model


class FeatureExtractor:
    def __init__(self, config:Config):
        self.config = config
        self.label, self.text, self.image = self._load_mat_file()
        self.model = self._load_model()
    

    def _load_mat_file(self):
        data_path = config['base_dir'] + config['data_name']

        label_file = data_path + config['label_file']
        text_file = data_path + config['text_file']
        image_file = data_path + config['image_file']

        label = loadmat(label_file)
        text = h5py.File(text_file)
        image = h5py.File(image_file)

        log('load mat file done')
        log('keys : ')
        log(label.keys())
        log(text.keys())
        log(image.keys())
        return label, text, image  

    def _load_model(self):
        model_name = self.config['model_name']

        if model_name == 'vgg19':
            vgg = VGG19(weights='imagenet')
            vgg = Model(vgg.input, vgg.get_layer('fc2').output)
            return vgg
        else:
            raise ValueError('unkown model name')
    
    def extract(self):
        labels = self.label['LALL']
        texts = self.text['YALL']
        images = self.image['IALL']

        all_img = []

        for e in range(len(images)):
            img = images[e].T
            log(img.shape)
            img = np.expand_dims(img, axis = 0)
            all_img.append(self.model.predict(img))
        
        all_img = np.vstack(all_img)
        log('images shape' , all_img.shape)

        data_path = self.config['base_dir'] + self.config['data_name']

        np.save(path + self.config['data_name']+'.npy', all_img)

        log('image save successfully')

        text_np = np.asarray(texts)
        log('text shape', text_np.shape)
        label_np = np.asarray(labels)
        log('label shape', label_np.shape)
        np.save(data_path + '/text.npy', text_np)
        np.save(data_path + '/label.npy', label_np)
        log('text and label save successfully')

    def split(self):
        log('split data')
        pass