import torch
import numpy as np
from torch.autograd import Variable


class Metricer:
    def __init__(self, config):
        self.config = config
    
    def _compress(self, train_loader, test_loader, model_I, model_T):
        re_BI = list([])
        re_BT = list([])
        re_L = list([])
        for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
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
        for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
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

            count = np.linspace(1, tsum, tsum)
            tindex = np.asarray(np.where(gnd == 1)) + 1.0
            map_ = np.mean(count / (tindex))
            map = map + map_
        map = map / num_query
        return map, large_hamming_samples

    def show_large_hamming_samples(self, large_hamming_samples, query_loader, database_loader):
        """
        展示汉明距离过大的样本
        :param large_hamming_samples: 汉明距离过大的样本列表
        :param query_loader: 查询数据加载器
        :param database_loader: 数据库数据加载器
        """
        if not large_hamming_samples:
            print("没有找到汉明距离过大的样本")
            return

        print("\n汉明距离过大的样本:")
        print("-" * 50)
        
        for sample in large_hamming_samples:
            query_idx = sample['query_index']
            retrieval_indices = sample['retrieval_indices']
            hamming_distances = sample['hamming_distances']
            
            # 获取查询样本
            query_data = query_loader.dataset[query_idx]
            query_img, query_txt, query_label, _ = query_data
            
            print(f"\n查询样本 {query_idx}:")
            print(f"查询标签: {query_label}")
            print(f"查询文本特征形状: {query_txt.shape}")
            print(f"查询图像特征形状: {query_img.shape}")
            
            print("\n对应的检索样本:")
            for i, (ret_idx, hamming_dist) in enumerate(zip(retrieval_indices, hamming_distances)):
                if i >= 5:  # 只显示前5个
                    break
                ret_data = database_loader.dataset[ret_idx]
                ret_img, ret_txt, ret_label, _ = ret_data
                print(f"\n检索样本 {ret_idx}:")
                print(f"汉明距离: {hamming_dist:.4f}")
                print(f"检索标签: {ret_label}")
                print(f"检索文本特征形状: {ret_txt.shape}")
                print(f"检索图像特征形状: {ret_img.shape}")
            print("-" * 50)