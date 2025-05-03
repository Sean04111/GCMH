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

    def eval_mAP_all(self, query_HashCode, retrieval_HashCode, query_Label, retrieval_Label, verbose=False):
        num_query = query_Label.shape[0]
        total_ap = 0

        all_positive_samples = []

        for iter in range(num_query):
            gnd = (np.dot(query_Label[iter, :], retrieval_Label.transpose()) > 0).astype(np.float32)
            tsum = int(np.sum(gnd))
            if tsum == 0:
                continue

            hamm = self.calculate_hamming(query_HashCode[iter, :], retrieval_HashCode)

            # 正样本索引
            positive_indices = np.where(gnd == 1)[0]

            # 排序索引（从小到大）
            sorted_indices = np.argsort(hamm)
            sorted_gnd = gnd[sorted_indices]
            sorted_hamm = hamm[sorted_indices]

            # 可选：打印排序列表及正负情况
            if verbose:
                print(f"\n=== Query {iter} ===")
                print(f"Query Image Name: {self.qurey_img_names[iter]}")
                print(f"Query Text: {self.qurey_raw_texts[iter]}")
                print("Top-20 Retrieval Results:")
                for rank, (idx, h_dist, is_pos) in enumerate(zip(sorted_indices[:20], sorted_hamm[:20], sorted_gnd[:20])):
                    symbol = "✔" if is_pos else "✘"
                    print(f"  Rank {rank+1:2d}: Index={idx:4d}, Hamming={h_dist:.1f}, Match={symbol}")

            # 收集正样本信息
            for pos_idx in positive_indices:
                all_positive_samples.append({
                    'hamming_distance': hamm[pos_idx],
                    'query_img_name': self.qurey_img_names[iter],
                    'query_text': self.qurey_raw_texts[iter]
                })

            # 计算 AP
            count = np.linspace(1, tsum, tsum)
            tindex = np.asarray(np.where(sorted_gnd == 1)) + 1.0
            ap = np.mean(count / (tindex))
            total_ap += ap

        mAP = total_ap / num_query
        return mAP, all_positive_samples

