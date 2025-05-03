import torch
import numpy as np
from torch.autograd import Variable

from utils.logger import log


class Metricer:
    def __init__(self, config, qurey_img_names, qurey_raw_texts):
        self.config = config
        self.qurey_img_names = qurey_img_names
        self.qurey_raw_texts = qurey_raw_texts
    
    def _compress(self, database_loader, query_loader, model_I, model_T):
        re_BI = list([])
        re_BT = list([])
        re_L = list([])
        for _, (data_I, data_T, data_L, _,_) in enumerate(database_loader):
            with torch.no_grad():
                var_data_I = Variable(data_I.cuda())
                _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            re_BI.extend(code_I.cpu().data.numpy())

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            re_BT.extend(code_T.cpu().data.numpy())
            re_L.extend(data_L.cpu().data.numpy())
        qu_BI = list([])
        qu_BT = list([])
        qu_L = list([])
        for idx, (data_I, data_T, data_L, _,_) in enumerate(query_loader):
            with torch.no_grad():
                var_data_I = Variable(data_I.cuda())
                _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            qu_BI.extend(code_I.cpu().data.numpy())

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            qu_BT.extend(code_T.cpu().data.numpy())
            qu_L.extend(data_L.cpu().data.numpy())
        re_BI = np.array(re_BI)
        re_BT = np.array(re_BT)
        re_L = np.array(re_L)

        qu_BI = np.array(qu_BI)
        qu_BT = np.array(qu_BT)
        qu_L = np.array(qu_L)
        return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

    def calculate_hamming(self, B1, B2):  # 添加 self 参数
        leng = B2.shape[1]
        distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
        return distH

    def eval_mAP_all(self, query_HashCode, retrieval_HashCode, query_Label, retrieval_Label):
        num_query = query_Label.shape[0]
        map = 0
        large_hamming_samples = []  # 存储汉明距离过大的样本
        
        # 存储所有样本的汉明距离信息
        all_hamming_samples = []
        
        for iter in range(num_query):
            gnd = (np.dot(query_Label[iter, :], retrieval_Label.transpose()) > 0).astype(np.float32)
            tsum = int(np.sum(gnd))
            if tsum == 0:
                continue
            hamm = self.calculate_hamming(query_HashCode[iter, :], retrieval_HashCode)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            # 记录汉明距离过大的样本
            large_hamming_indices = np.where(hamm > np.mean(hamm) + 2 * np.std(hamm))[0]
            if len(large_hamming_indices) > 0:
                large_hamming_samples.append({
                    'query_index': iter,
                    'retrieval_indices': large_hamming_indices,
                    'hamming_distances': hamm[large_hamming_indices]
                })

            # 记录所有样本的汉明距离
            for ret_idx, hamming_dist in enumerate(hamm):
                all_hamming_samples.append({
                    'query_index': iter,
                    'retrieval_index': ret_idx,
                    'hamming_distance': hamming_dist,
                    'query_img_name': self.qurey_img_names[iter],
                    'query_text': self.qurey_raw_texts[iter]
                })

            count = np.linspace(1, tsum, tsum)
            tindex = np.asarray(np.where(gnd == 1)) + 1.0
            map_ = np.mean(count / (tindex))
            map = map + map_
        map = map / num_query

        # 按汉明距离排序并输出前n个
        all_hamming_samples.sort(key=lambda x: x['hamming_distance'], reverse=True)

        top_hamming_samples = all_hamming_samples[:self.config['top_hamming']]
        return map, top_hamming_samples
