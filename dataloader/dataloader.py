from cProfile import label
from utils.logger import log
from random import shuffle
from re import I
from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


class BatchData(Dataset):
    def __init__(self, images, texts, labels, img_names, raw_texts):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.img_names = img_names
        self.raw_texts = raw_texts
        
    def __getitem__(self, index):
        image = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        img_name = self.img_names[index]
        raw_text = self.raw_texts[index]
        return image, text, label, img_name, raw_text

    def __len__(self):
        length = len(self.images)
        # 强制要求images和labels的长度相等
        assert len(self.images) ==  len(self.labels)
        return length

    def get_all_data(self):
        return self.images, self.texts, self.labels, self.img_names, self.raw_texts


class CustomDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataloaders = {}
        self._load_data()

    def __getitem__(self, key):
        return self.dataloaders[key]

    def _load_data(self):
        base_path = self.config['base_path'] + self.config['data_name'] + '/'
        npy_name = self.config['data_name'] + '_' + self.config['model_name'] + '.npy'

        images = np.load(base_path + npy_name)
        tags = np.load(base_path + 'tags.npy')
        label = np.load(base_path + 'label.npy')

        test_num = np.load(base_path + 'test_num.npy')
        val_num = np.load(base_path + 'val_num.npy')
        train_num = np.load(base_path + 'train_num.npy')
        database_num = np.load(base_path + 'database_num.npy')

        # 类型转换
        images = images.astype(np.float32)
        tags = tags.astype(np.float32)
        label = label.astype(int)

        train_images = images[train_num]
        train_texts = tags[train_num]
        train_labels = label[train_num]

        validation_images = images[val_num]
        validation_texts = tags[val_num]
        validation_labels = label[database_num]

        # todo: 这里直接重新命名为query了，后面要统一一下名字(4/28)
        query_images = images[test_num]
        query_texts = tags[test_num]
        query_labels = label[test_num]

        database_images = images[database_num]
        database_texts = tags[database_num]
        database_labels = label[database_num]

        # todo: 后续看下database_v

        if self.config['data_name'] == 'nus':
            dataset = {
                'train': BatchData(images=train_images, texts=train_texts, labels=train_labels, img_names = {}, raw_texts = {}),
                'query': BatchData(images=query_images, texts=query_texts, labels=query_labels, img_names = {}, raw_texts = {}),
                'validation': BatchData(images=validation_images, texts=validation_texts, labels=validation_labels, img_names = {}, raw_texts = {}),
                'database': BatchData(images=database_images, texts=database_texts, labels=database_labels, img_names = {}, raw_texts = {}),
            }
        elif self.config['data_name'] == 'flickr':
            #  load img_names and raw_texts
            img_names = np.load(base_path + 'img_names.npy')
            raw_texts = np.load(base_path + 'raw_texts.npy')

            dataset = {
                'train': BatchData(images=train_images, texts=train_texts, labels=train_labels, img_names = img_names, raw_texts = raw_texts),
                'query': BatchData(images=query_images, texts=query_texts, labels=query_labels, img_names = img_names, raw_texts = raw_texts),
                'validation': BatchData(images=validation_images, texts=validation_texts, labels=validation_labels, img_names = img_names, raw_texts = raw_texts),
                'database': BatchData(images=database_images, texts=database_texts, labels=database_labels, img_names = img_names, raw_texts = raw_texts),    
            }

        self.dataloaders = {
            'train': DataLoader(dataset['train'], batch_size=self.config['batch_size'], shuffle=True, num_workers=4),
            'query': DataLoader(dataset['query'], batch_size=self.config['batch_size'], shuffle=False, num_workers=4),
            'validation': DataLoader(dataset['validation'], batch_size=self.config['batch_size'], shuffle=False, num_workers=4),
            'database': DataLoader(dataset['database'], batch_size=self.config['batch_size'], shuffle=False, num_workers=4),
        }
        
        log('load data done')




    
 

       