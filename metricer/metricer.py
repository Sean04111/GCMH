import torch
import numpy as np
from torch.autograd import Variable
from utils.logger import log
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt



class Metricer:
    def __init__(self, config, qurey_img_names, qurey_raw_texts,database_img_names, database_raw_texts):
        self.config = config
        self.qurey_img_names = qurey_img_names
        self.qurey_raw_texts = qurey_raw_texts
        self.database_img_names = database_img_names
        self.database_raw_texts = database_raw_texts
    
    def _compress(self, database_loader, query_loader, model_I, model_T):
        re_BI = list([])
        re_BT = list([])
        re_L = list([])
        for _, data in enumerate(database_loader):

            if self.config['data_name'] == 'old_flickr':
               data_I, data_T, data_L, _,_, _ = data
            else:
                data_I, data_T, data_L, _ = data
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
        for idx, data in enumerate(query_loader):

            if self.config['data_name'] == 'old_flickr':
                data_I, data_T, data_L, _,_, _ = data
            else:
                data_I, data_T, data_L, _ = data
            
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

    def eval_mAP_all(self, query_HashCode, retrieval_HashCode, query_Label, retrieval_Label,
                     epoch_num, query_type, outdata_type, verbose=False):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # === 数据搬到 GPU ===
        query_HashCode = torch.from_numpy(query_HashCode).float().to(device)
        retrieval_HashCode = torch.from_numpy(retrieval_HashCode).float().to(device)
        query_Label = torch.from_numpy(query_Label).float().to(device)
        retrieval_Label = torch.from_numpy(retrieval_Label).float().to(device)
    
        num_query = query_Label.shape[0]
        total_ap = 0.0
        num_valid_queries = 0
    
        all_sorted_gnds = []
        max_recall_length = 0
    
        for i in range(num_query):
            q_label = query_Label[i]  # [L]
            gnd = torch.matmul(q_label, retrieval_Label.T) > 0  # [N]
            tsum = torch.sum(gnd).item()
            if tsum == 0:
                continue
    
            q_code = query_HashCode[i]  # [n_bits]
            hamm = (q_code != retrieval_HashCode).sum(dim=1).float()  # 汉明距离 [N]
            sorted_indices = torch.argsort(hamm)  # [N]
            sorted_gnd = gnd[sorted_indices].float()  # [N]
            max_recall_length = max(max_recall_length, int(torch.sum(sorted_gnd).item()))
            all_sorted_gnds.append(sorted_gnd.cpu().numpy())  # for PR curve
    
            if verbose and i == num_query - 1 and self.config['data_name'] == 'old_flickr':
                print(f"\n=== Query {i} ===")
                print(f"Query Image Name: {self.qurey_img_names[i]}")
                print(f"Query Text: {self.qurey_raw_texts[i]}")
                print("Top-10 Retrieval Results:")
                for rank, idx in enumerate(sorted_indices[:10]):
                    h_dist = hamm[idx].item()
                    is_pos = sorted_gnd[rank].item()
                    symbol = "✔" if is_pos else "✘"
                    img_name = self.database_img_names[idx] if idx < len(self.database_img_names) else "unknown"
                    text = self.database_raw_texts[idx] if idx < len(self.database_raw_texts) else "unknown"
                    print(f"  Rank {rank+1:2d}: Match={symbol} Image={img_name:15s}, Text={text:15s}, Hamming={h_dist:.1f}")
    
            pos_indices = torch.nonzero(sorted_gnd).squeeze() + 1.0  # [tsum]
            count = torch.linspace(1, tsum, steps=tsum).to(device)
            ap = torch.mean(count / pos_indices[:tsum])
            total_ap += ap.item()
            num_valid_queries += 1
    
        mAP = total_ap / num_valid_queries
    
        # === 哈希码 bit 熵计算 ===
        probs = torch.mean((query_HashCode > 0).float(), dim=0)
        entropies = -probs * torch.log2(probs + 1e-10) - (1 - probs) * torch.log2(1 - probs + 1e-10)
        mean_entropy = entropies.mean().item()
    
        # === 查询热度图（使用 CPU） ===
        if verbose and epoch_num > 20 and outdata_type == 'heat':
            log("retrivel result gen start")
            query_scores = []
    
            for i in tqdm(range(num_query), desc="Computing query heat scores"):
                q_label = query_Label[i]
                gnd = torch.matmul(q_label, retrieval_Label.T) > 0
                if torch.sum(gnd).item() == 0:
                    query_scores.append(0.0)
                    continue
    
                q_code = query_HashCode[i]
                hamm = (q_code != retrieval_HashCode).sum(dim=1).float()
                sorted_indices = torch.argsort(hamm)
    
                score = 0.0
                for rank, db_idx in enumerate(sorted_indices):
                    overlap = torch.sum(query_Label[i] * retrieval_Label[db_idx])
                    score += overlap.item() / (rank + 1)
                query_scores.append(score)
    
            query_scores = np.array(query_scores)
            query_scores = (query_scores - np.min(query_scores)) / (np.max(query_scores) - np.min(query_scores) + 1e-10)
    
            reduced = TSNE(n_components=2, random_state=0).fit_transform(query_HashCode.cpu().numpy())
    
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=query_scores, cmap='hot_r', s=35)
            plt.colorbar(label="Query Heat Score")
            plt.title("Query Result Hotness Visualization (Red = Better)")
            plt.tight_layout()
            plt.savefig("query_heatmap_" + query_type + ".png")
            plt.close()
    
        # === 绘制 PR 曲线（仍使用 CPU） ===
        if verbose and epoch_num > 20 and outdata_type == 'pr':
            log('绘制pr曲线...')
            interpolated_recalls = np.linspace(0, 1, 100)
            interpolated_precisions = []
    
            for sorted_gnd in tqdm(all_sorted_gnds):
                precisions = []
                recalls = []
                tp = 0
                total_pos = np.sum(sorted_gnd)
                for i, rel in enumerate(sorted_gnd):
                    if rel == 1:
                        tp += 1
                    precision = tp / (i + 1)
                    recall = tp / (total_pos + 1e-10)
                    precisions.append(precision)
                    recalls.append(recall)
                interpolated = np.interp(interpolated_recalls, recalls, precisions, left=precisions[0], right=precisions[-1])
                interpolated_precisions.append(interpolated)
    
            avg_precision = np.mean(interpolated_precisions, axis=0)
            plt.figure()
            plt.plot(interpolated_recalls, avg_precision, label='Average PR Curve')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Mean PR Curve @epoch %d" % epoch_num)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("pr_curve_" + query_type + ".png")
            plt.close()
    
        return mAP, mean_entropy



    def compute_bit_entropy(self, hash_codes):
        """
        计算每一位哈希码的熵，输入为 shape [N, K]，N 是样本数，K 是哈希位数
        哈希值应该为 ±1
        """
        # 转换为 [0, 1] 表示（-1→0, +1→1）
        binary_codes = (hash_codes > 0).astype(np.int32)  # shape: [N, K]
        
        entropies = []
        for bit in range(binary_codes.shape[1]):
            p = np.mean(binary_codes[:, bit])  # 是 1 的概率
            if p == 0 or p == 1:
                entropy = 0.0
            else:
                entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        return entropies








