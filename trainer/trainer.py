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
import datetime
from utils.logger import log
from tqdm import tqdm 



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
        _,_,_, self.database_img_names, self.database_raw_texts = self.database_loader.dataset.get_all_data()

        self.best_mAP = 0.0
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
        self.save_file_name = self.config['result_path'] + formatted_time + '.txt'
        

        self.metricer = Metricer(config=self.config, qurey_img_names=self.query_img_names, qurey_raw_texts=self.query_raw_texts,database_img_names=self.database_img_names, database_raw_texts=self.database_raw_texts)

        train_imgs = F.normalize(torch.Tensor(train_imgs).cuda())

        # 对clip做特殊化处理
        if self.config['model_name'] == 'clip':
            img_tensor = torch.Tensor(train_imgs)
            train_imgs = (img_tensor - img_tensor.mean(dim=0)) / (img_tensor.std(dim=0) + 1e-6)
            train_imgs = train_imgs.cuda()
        else:
            train_imgs = F.normalize(torch.Tensor(train_imgs).cuda())
            
        train_txts = F.normalize(torch.Tensor(train_txts).cuda())
        train_labels = torch.Tensor(train_labels).cuda()

        self.Sgc, self.distance_matrix, self.possibility_matrix = self._build_similarity_matrix(train_imgs, train_txts)

        txt_feat_len = train_txts.size(1)
        self.ImgNet = ImgNet(code_len=self.config['bit'], txt_feat_len=txt_feat_len)
        self.TxtNet = TxtNet(code_len=self.config['bit'], txt_feat_len=txt_feat_len)

        self.opt_Img = torch.optim.SGD(self.ImgNet.parameters(), lr = self.config['learning_rate'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
        self.opt_Txt = torch.optim.SGD(self.TxtNet.parameters(), lr = self.config['learning_rate'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])

        print("=== 标签检查 ===")
        label_sum = train_labels.sum(dim=1).cpu().numpy()
        print(f"训练集中每个样本平均标签数: {np.mean(label_sum):.2f}")
        print(f"标签为全 0 的样本数: {np.sum(label_sum == 0)} / {len(label_sum)}")
        print("=== 标签类别分布 ===")
        label_distribution = train_labels.sum(dim=0).cpu().numpy()
        print(f"每个类别对应的样本数: {label_distribution}")
        print(f"类别最大/最小样本数: {label_distribution.max()} / {label_distribution.min()}")
        print("=== 图像/文本特征分布 ===")
        print("图像特征均值/方差:", train_imgs.mean().item(), train_imgs.std().item())
        print("文本特征均值/方差:", train_txts.mean().item(), train_txts.std().item())
        print("训练样本数量:", len(train_imgs))
        print("训练集 loader 长度:", len(self.train_loader))
        print("Query loader 长度:", len(self.query_loader))

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
        norm_HashCode_Img = F.normalize(HashCode_Img)
        norm_HashCode_Txt = F.normalize(HashCode_Txt)
    
        I_I = cosine_similarity(norm_HashCode_Img, norm_HashCode_Img)
        T_T = cosine_similarity(norm_HashCode_Txt, norm_HashCode_Txt)
        I_T = cosine_similarity(norm_HashCode_Img, norm_HashCode_Txt)
        T_I = cosine_similarity(norm_HashCode_Txt, norm_HashCode_Img)
    
        diagonal = I_T.diagonal()
        all_1 = torch.ones_like(diagonal)
        loss_pair = F.mse_loss(diagonal, self.config['K'] * all_1)
    
        loss_dis_1 = F.mse_loss(T_T * (1-I), Sgc * (1-I))
        loss_dis_2 = F.mse_loss(I_T * (1-I), Sgc * (1-I))
        loss_dis_3 = F.mse_loss(I_I * (1-I), Sgc * (1-I))
        loss_g = (loss_dis_1 + loss_dis_2 + loss_dis_3) * self.config['dw']
    
        loss_cons = (F.mse_loss(I_T, I_I) + F.mse_loss(I_T, T_T) + F.mse_loss(I_I, T_T) + F.mse_loss(I_T, T_I)) * self.config['cw']
    
        loss = loss_pair + loss_g + loss_cons
    
        return loss, loss_pair, loss_g, loss_cons


    # 计算mAP@ALL
    def _eval(self, epoch_num):
        self.ImgNet.eval().cuda()
        self.TxtNet.eval().cuda()
        re_HashCode_Img, re_HashCode_Txt, re_Label, qu_HashCode_Img, qu_HashCode_Txt, qu_Label = self.metricer._compress(self.database_loader, self.query_loader, self.ImgNet, self.TxtNet)

        # entropies = self.metricer.compute_bit_entropy(qu_HashCode_Img)
        # print("\n=== img 查询集 哈希码熵 分布 ===")
        # print(f"Mean Entropy: {np.mean(entropies):.4f}")

        # entropies = self.metricer.compute_bit_entropy(qu_HashCode_Txt)
        # print("\n=== txt 查询集 哈希码熵 分布 ===")
        # print(f"Mean Entropy: {np.mean(entropies):.4f}")

        mAP_I2T , entropies_Q_img = self.metricer.eval_mAP_all(query_HashCode=qu_HashCode_Img, retrieval_HashCode=re_HashCode_Txt, query_Label=qu_Label, retrieval_Label=re_Label,epoch_num=epoch_num, query_type='img',outdata_type = 'heat', verbose=True)
        mAP_T2I, entropies_Q_txt = self.metricer.eval_mAP_all(query_HashCode=qu_HashCode_Txt, retrieval_HashCode=re_HashCode_Img, query_Label=qu_Label, retrieval_Label=re_Label,epoch_num=epoch_num, query_type='txt', outdata_type='heat', verbose=True)

        return mAP_I2T, mAP_T2I, entropies_Q_img, entropies_Q_txt

    def _save_best_result(self, mAP_I2T, mAP_T2I):
        with open(self.save_file_name, 'w') as f:
            f.write("config is "+ str(self.config.config) + "\n")
            mAP_text = "best mAP_I2T : "  + str(mAP_I2T) + " best mAP_T2I : " + str(mAP_T2I)
            f.write(mAP_text)

    # 自训练
    def train(self):
        epoch_num = self.config['epoch_num']
        
        # === 初始化 CSV 日志文件 ===
        with open("loss_log.csv", "w") as f:
            f.write("epoch,total_loss,loss_pair,loss_g,loss_cons\n")
    
        for epoch in range(epoch_num):
            self.ImgNet.cuda().train()
            self.TxtNet.cuda().train()
    
            # 每轮累计值
            sum_loss, sum_pair, sum_g, sum_cons = 0.0, 0.0, 0.0, 0.0
    
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epoch_num}")
            for idx, data in pbar:
                if self.config['data_name'] == 'old_flickr':
                    img, txt, labels, img_name, raw_text, index = data
                else:
                    img, txt, labels, index = data
    
                img = Variable(img).cuda()
                txt = Variable(torch.FloatTensor(txt.numpy())).cuda()
    
                batch_size = img.size(0)
                I = torch.eye(batch_size).cuda()
                _, HashCode_Img = self.ImgNet(img)
                _, HashCode_Txt = self.TxtNet(txt)
                Sgc = self.Sgc[index, :][:, index].cuda()
    
                # === 一阶段完整更新 ===
                loss, loss_pair, loss_g, loss_cons = self._loss_cal(HashCode_Img, HashCode_Txt, Sgc, I)
    
                self.opt_Img.zero_grad()
                self.opt_Txt.zero_grad()
                loss.backward()
                self.opt_Img.step()
                self.opt_Txt.step()
    
                # === 单独优化图像网络 ===
                _, HashCode_Img = self.ImgNet(img)
                _, HashCode_Txt = self.TxtNet(txt)
                loss_img, _, _, _ = self._loss_cal(HashCode_Img, HashCode_Txt.sign().detach(), Sgc, I)
                self.opt_Img.zero_grad()
                loss_img.backward()
                self.opt_Img.step()
    
                # === 单独优化文本网络 ===
                loss_txt, _, _, _ = self._loss_cal(HashCode_Img.sign().detach(), HashCode_Txt, Sgc, I)
                self.opt_Txt.zero_grad()
                loss_txt.backward()
                self.opt_Txt.step()
    
                # === 累加统计值 ===
                sum_loss += loss.item()
                sum_pair += loss_pair.item()
                sum_g += loss_g.item()
                sum_cons += loss_cons.item()
    
                avg_loss = sum_loss / (idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
            # === 每轮平均值 ===
            avg_loss = sum_loss / len(self.train_loader)
            avg_pair = sum_pair / len(self.train_loader)
            avg_g = sum_g / len(self.train_loader)
            avg_cons = sum_cons / len(self.train_loader)
    
            # === 打印 + 写入 CSV ===
            log(f"[Epoch {epoch}] Total: {avg_loss:.4f} | Lpair: {avg_pair:.4f} | Lg: {avg_g:.4f} | Lcons: {avg_cons:.4f}")
            with open("loss_log.csv", "a") as f:
                f.write(f"{epoch},{avg_loss:.4f},{avg_pair:.4f},{avg_g:.4f},{avg_cons:.4f}\n")
    
            # === mAP 验证阶段 ===
            if epoch >= self.config['eval_epoch']:
                mAP_I2T, mAP_T2I, e_q_i, e_q_t = self._eval(epoch)
                log('Epoch: {}, Loss: {:.4f}, mAP_I2T: {:.4f}, mAP_T2I: {:.4f}, entropies_query_image: {:.4f}, entropies_query_text: {:.4f}'.format(
                    epoch, avg_loss, mAP_I2T, mAP_T2I, e_q_i, e_q_t))
    
                if mAP_I2T + mAP_T2I > self.best_mAP:
                    log('logging the best score...')
                    self.best_mAP = mAP_I2T + mAP_T2I
                    self._save_best_result(mAP_I2T, mAP_T2I)

            
