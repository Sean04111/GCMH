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


class PreProcessor:
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

    # 加载模型
    def _load_model(self):
        model_name = self.config['model_name']

        if model_name == 'vgg19':
            vgg = VGG19(weights='imagenet')
            vgg = Model(vgg.input, vgg.get_layer('fc2').output)
            return vgg
        else:
            raise ValueError('unkown model name')
    
    def wirfickr(self):
        pass
    def nus_wide(self):
        pass
    # 提取特征
    def extract(self):
        if self.config['data_name'] == 'nus':
            labels = self.label['LAll']
            texts = self.text['YAll']
            images = self.image['IAll']

            

        all_img = []

        for e in range(len(images)):
            img = images[e].T
            log(img.shape)
            img = np.expand_dims(img, axis = 0)
            all_img.append(self.model.predict(img))
        
        all_img = np.vstack(all_img)
        log('images shape' , all_img.shape)

        data_path = self.config['base_dir'] + self.config['data_name']

        np.save(path + self.config['data_name']+'_'+self.config['model_name']+'.npy', all_img)

        log('image save successfully')

        text_np = np.asarray(texts)
        log('text shape', text_np.shape)
        label_np = np.asarray(labels)
        log('label shape', label_np.shape)
        np.save(data_path + '/text.npy', text_np)
        np.save(data_path + '/label.npy', label_np)
        log('text and label save successfully')

    # 划分数据集 
    def split(self):
        data_path = self.config['base_dir'] + self.config['data_name']
        try:
            labels = np.load(data_path + './label.npy')
        except FileNotFoundError:
            log(data_path + './label.npt not found')
            return
        
        N = labels.shape[0]
        all_indices = np.arange(N)
        
        np.random.seed(42)
        np.random.shuffle(all_indices)

        database_num = N - self.config['test_num']
        test_num = self.config['test_num']
        val_num = self.config['val_num']
        train_num = self.config['train_num']

      

        # 划分训练集和测试集
        test_idx = all_indices[:test_num]
        val_idx = all_indices[test_num:test_num+val_num]
        train_idx = all_indices[test_num+val_num:test_num+val_num+train_num]
        database_idx = all_indices[test_num:]

        np.save(data_path + 'test_num.npy', test_idx)
        np.save(data_path + 'val_num.npy', val_idx)
        np.save(data_path + 'train_num.npy', train_idx)
        np.save(data_path + 'database_num.npy', database_idx)

        log('data split done')
        log('test num', test_num)
        log('val num', val_num)
        log('train num', train_num)
        log('database num', database_num)