import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config.config import Config
from dataloader.dataloader import CustomDataLoader
from utils.similarity import cosine_similarity
from models.ImgNet import ImgNet
from models.TxtNet import TxtNet
from metricer.metricer import Metricer
from utils.logger import log



class Trainer:
    def __init__(self, config_path):
        self.config = Config(config_path=config_path)
        dataloader = CustomDataLoader(config=self.config)

        self.train_loader = dataloader['train']
        self.val_loader = dataloader['validation']
        self.query_loader = dataloader['query']
        self.database_loader = dataloader['database']

        train_imgs,train_txts,train_labels,_, _ = self.train_loader.dataset.get_all_data()
        _, _, _, self.query_img_names, self.query_raw_texts = self.query_loader.dataset.get_all_data()

        self.metricer = Metricer(config=self.config, qurey_img_names=self.query_img_names, qurey_raw_texts=self.query_raw_texts)

        train_imgs = F.normalize(torch.Tensor(train_imgs).cuda())
        train_txts = F.normalize(torch.Tensor(train_txts).cuda())
        train_labels = torch.Tensor(train_labels).cuda()

        self.Sgc, self.distance_matrix, self.possibility_matrix = self._build_similarity_matrix(train_imgs, train_txts)

        txt_feat_len = train_txts.size(1)
        self.ImgNet = ImgNet(code_len=self.config['bit'], txt_feat_len=txt_feat_len)
        self.TxtNet = TxtNet(code_len=self.config['bit'], txt_feat_len=txt_feat_len)

        self.opt_Img = torch.optim.SGD(self.ImgNet.parameters(), lr = self.config['learning_rate'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
        self.opt_Txt = torch.optim.SGD(self.TxtNet.parameters(), lr = self.config['learning_rate'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])

    # 构建相似度矩阵
    def _build_similarity_matrix(self, images_feature, texts_feature):
        batch_size = images_feature.size(0)
        # 计算自相似度矩阵
        images_feature = F.normalize(images_feature)
        images_self_similarity = cosine_similarity(images_feature, images_feature)
        texts_feature = F.normalize(texts_feature)
        texts_self_similarity = cosine_similarity(texts_feature, texts_feature)

        # 计算距离矩阵
        distance_weight = self.config['distance_weight']
        distance_matrix = (1-distance_weight) * images_self_similarity + distance_weight * texts_self_similarity

        # 基于k近邻计算两节点相似度
        K = self.config['KNN']
        distance_matrix_v2 = distance_matrix.clone()
        m, n1 = distance_matrix_v2.sort()
        k_rows = torch.arange(batch_size).view(-1,1).repeat(1, K).view(-1)
        k_cols = n1[:, :K].contiguous().view(-1)
        distance_matrix_v2[k_rows, k_cols] = 0.

        top_rows = torch.arange(batch_size).view(-1)
        top_cols = n1[:, -1:].contiguous().view(-1)
        distance_matrix_v2[top_rows, top_cols] = 0.

        distance_matrix_v2 = distance_matrix_v2 / distance_matrix_v2.sum(1).view(-1,1)
        p = cosine_similarity(distance_matrix_v2, distance_matrix_v2)

        S = (1-self.config['possibility_weight']) * distance_matrix + self.config['possibility_weight'] * self.config['possibility_scale'] * p

        S = S * 2.0 - 1

        log("Similarity matrix built.")

        return S, distance_matrix, p

    def _loss_cal(self, HashCode_Img, HashCode_Txt, Sgc, I):
        HashCode_Img = F.normalize(HashCode_Img)
        HashCode_Txt = F.normalize(HashCode_Txt)

        I_I = cosine_similarity(HashCode_Img, HashCode_Img)
        T_T = cosine_similarity(HashCode_Txt, HashCode_Txt)
        I_T = cosine_similarity(HashCode_Img, HashCode_Txt)
        T_I = cosine_similarity(HashCode_Txt, HashCode_Img)

        diagonal = I_T.diagonal()
        all_1 = torch.rand(T_T.size(0)).fill_(1).cuda()
        loss_pair = F.mse_loss(diagonal, self.config['K'] * all_1)

        loss_dis_1 = F.mse_loss(T_T * (1-I), Sgc* (1-I))
        loss_dis_2 = F.mse_loss(I_T * (1-I), Sgc* (1-I))
        loss_dis_3 = F.mse_loss(I_I * (1-I), Sgc* (1-I))
        loss_dis_4 = F.mse_loss(T_I * (1-I), Sgc* (1-I))

        loss_cons = F.mse_loss(I_T, I_I) + F.mse_loss(I_T, T_T) + F.mse_loss(I_I, T_T) + F.mse_loss(I_T, T_I)

        loss = loss_pair + (loss_dis_1 + loss_dis_2 + loss_dis_3 ) * self.config['dw'] + loss_cons * self.config['cw']

        return loss

    # 计算mAP@ALL
    def _eval(self):
        re_HashCode_Img, re_HashCode_Txt, re_Label, qu_HashCode_Img, qu_HashCode_Txt, qu_Label = self.metricer._compress(self.database_loader, self.query_loader, self.ImgNet, self.TxtNet)
        mAP_I2T, large_hamming_I2T = self.metricer.eval_mAP_all(query_HashCode=qu_HashCode_Img, retrieval_HashCode=re_HashCode_Txt, query_Label=qu_Label, retrieval_Label=re_Label)
        mAP_T2I, large_hamming_T2I = self.metricer.eval_mAP_all(query_HashCode=qu_HashCode_Txt, retrieval_HashCode=re_HashCode_Img, query_Label=qu_Label, retrieval_Label=re_Label)

        log("I2T 任务，异常样本 : ", large_hamming_I2T)
        log("T2I 任务，异常样本 : ", large_hamming_T2I)
        
        return mAP_I2T, mAP_T2I

    # 自训练
    def train(self):
        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()
        epoch_num = self.config['epoch_num']
        
        for epoch in range(epoch_num):
            for idx, (img, txt, labels, img_name, raw_text) in enumerate(self.train_loader):
                img = Variable(img).cuda()
                txt = Variable(torch.FloatTensor(txt.numpy())).cuda()
                
                # 获取当前批次中每个样本的全局索引
                batch_indices = torch.arange(idx * self.config['batch_size'], 
                                          min((idx + 1) * self.config['batch_size'], 
                                              len(self.train_loader.dataset))).cuda()
                
                batch_size = img.size(0)
                I = torch.eye(batch_size).cuda()
                _, HashCode_Img = self.ImgNet(img)
                _, HashCode_Txt = self.TxtNet(txt)
                Sgc = self.Sgc[batch_indices, :][:, batch_indices].cuda()

                loss = self._loss_cal(HashCode_Img, HashCode_Txt, Sgc, I)

                self.opt_Img.zero_grad()
                self.opt_Txt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt_Img.step()
                self.opt_Txt.step()

                _, HashCode_Img = self.ImgNet(img)
                _, HashCode_Txt = self.TxtNet(txt)

                loss_img = self._loss_cal(HashCode_Img, HashCode_Txt.sign().detach(), Sgc, I)
                self.opt_Img.zero_grad()
                loss_img.backward(retain_graph=True)
                self.opt_Img.step()

                loss_txt = self._loss_cal(HashCode_Img.sign().detach(), HashCode_Txt, Sgc, I)
                self.opt_Txt.zero_grad()
                loss_txt.backward()
                self.opt_Txt.step()

                
                if (idx + 1) % len(self.train_loader) == 0:
                    mAP_I2T, mAP_T2I = self._eval()
                    log('Epoch: {}, Loss: {}, mAP_I2T: {}, mAP_T2I: {}'.format(epoch, loss, mAP_I2T, mAP_T2I))
            
