import torch
import numpy as np
from torch.autograd import Variable
from utils.logger import log
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


class Metricer:
    def __init__(self, config, qurey_img_names, qurey_raw_texts, database_img_names, database_raw_texts):
        self.config = config
        self.qurey_img_names = qurey_img_names
        self.qurey_raw_texts = qurey_raw_texts
        self.database_img_names = database_img_names
        self.database_raw_texts = database_raw_texts

    def _compress(self, database_loader, query_loader, model_I, model_T):
        def encode(loader):
            all_img, all_txt, all_label = [], [], []
            for data in loader:
                if self.config['data_name'] == 'old_flickr':
                    data_I, data_T, data_L, _, _, _ = data
                else:
                    data_I, data_T, data_L, _ = data

                with torch.no_grad():
                    data_I = data_I.cuda()
                    data_T = data_T.cuda() if isinstance(data_T, torch.Tensor) else torch.FloatTensor(data_T.numpy()).cuda()

                    _, code_I = model_I(data_I)
                    _, code_T = model_T(data_T)

                    code_I = torch.sign(code_I)
                    code_T = torch.sign(code_T)

                all_img.append(code_I)
                all_txt.append(code_T)
                all_label.append(data_L.cuda())

            return torch.cat(all_img), torch.cat(all_txt), torch.cat(all_label)

        re_BI, re_BT, re_L = encode(database_loader)
        qu_BI, qu_BT, qu_L = encode(query_loader)

        return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

    def calculate_hamming_gpu(self, B1, B2):
        # B1: [Q, K], B2: [D, K]
        K = B1.shape[1]
        sim = torch.matmul(B1, B2.T)
        return 0.5 * (K - sim)

    def eval_mAP_all(self, query_HashCode, retrieval_HashCode, query_Label, retrieval_Label,
                 epoch_num, query_type, outdata_type, verbose=False):
        device = query_HashCode.device
        Q = query_Label.shape[0]
        D = retrieval_Label.shape[0]
    
        # 保证所有数据都在同一设备上并为 float 类型
        query_HashCode = query_HashCode.to(device)
        retrieval_HashCode = retrieval_HashCode.to(device)
        query_Label = query_Label.to(device).float()
        retrieval_Label = retrieval_Label.to(device).float()
    
        hamm_dist = self.calculate_hamming_gpu(query_HashCode, retrieval_HashCode)  # [Q, D]
    
        total_ap = 0.0
        all_sorted_gnds = []
        max_recall_length = 0
    
        for i in range(Q):
            gnd = (torch.matmul(query_Label[i], retrieval_Label.T) > 0).float()  # [D]
            tsum = gnd.sum()
            if tsum == 0:
                continue
    
            hamm_row = hamm_dist[i]
            sorted_idx = torch.argsort(hamm_row)
            sorted_gnd = gnd[sorted_idx]
            max_recall_length = max(max_recall_length, int(sorted_gnd.sum().item()))
            all_sorted_gnds.append(sorted_gnd.cpu().numpy())
    
            # 打印前十样本
            if verbose and i == Q - 1 and self.config['data_name'] == 'old_flickr':
                print(f"\n=== Query {i} ===")
                print(f"Query Image Name: {self.qurey_img_names[i]}")
                print(f"Query Text: {self.qurey_raw_texts[i]}")
                print("Top-10 Retrieval Results:")
                for rank, idx in enumerate(sorted_idx[:10]):
                    is_pos = sorted_gnd[rank].item() == 1.0
                    symbol = "✔" if is_pos else "✘"
                    img_name = self.database_img_names[idx] if idx < len(self.database_img_names) else "unknown"
                    text = self.database_raw_texts[idx] if idx < len(self.database_raw_texts) else "unknown"
                    print(f"  Rank {rank+1:2d}: Match={symbol} Image={img_name:15s}, Text={text:15s}, Hamming={hamm_row[idx].item():.1f}")
    
            relevant = (sorted_gnd == 1).nonzero().squeeze()
            if relevant.numel() == 0:
                continue
            precision_at_k = torch.arange(1, relevant.numel() + 1, dtype=torch.float32).cuda() / (relevant + 1).float()
            ap = precision_at_k.mean().item()
            total_ap += ap
    
        mAP = total_ap / Q
    
        # === 熵 ===
        entropies = self.compute_bit_entropy(query_HashCode.cpu().numpy())
    
        # === 可视化 ===
        if verbose and epoch_num > 20 and outdata_type == 'heat':
            log("retrivel result gen start")
            self._draw_query_heatmap(query_HashCode.cpu().numpy(), query_Label.cpu().numpy(), retrieval_Label.cpu().numpy(), query_type)
    
        if verbose and epoch_num > 20 and outdata_type == 'pr':
            self._draw_pr_curve(all_sorted_gnds, query_type, epoch_num)
    
        return mAP, np.mean(entropies)


    def _draw_query_heatmap(self, query_HashCode, query_Label, retrieval_Label, query_type):
        query_scores = []
        for i in tqdm(range(query_Label.shape[0]), desc="Computing query heat scores"):
            gnd = (np.dot(query_Label[i], retrieval_Label.T) > 0).astype(np.float32)
            if np.sum(gnd) == 0:
                query_scores.append(0.0)
                continue

            hamm = 0.5 * (query_HashCode.shape[1] - np.dot(query_HashCode[i], retrieval_Label.T))
            sorted_indices = np.argsort(hamm)
            score = 0.0
            for rank, db_idx in enumerate(sorted_indices):
                overlap = np.sum(query_Label[i] * retrieval_Label[db_idx])
                score += overlap / (rank + 1)
            query_scores.append(score)

        query_scores = np.array(query_scores)
        query_scores = (query_scores - np.min(query_scores)) / (np.max(query_scores) - np.min(query_scores) + 1e-10)
        reduced = TSNE(n_components=2, random_state=0).fit_transform(query_HashCode)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=query_scores, cmap='hot_r', s=35)
        plt.colorbar(label="Query Heat Score")
        plt.title("Query Result Hotness Visualization (Red = Better)")
        plt.tight_layout()
        plt.savefig("query_heatmap_" + query_type + ".png")
        plt.close()

    def _draw_pr_curve(self, all_sorted_gnds, query_type, epoch_num):
        log('绘制pr曲线...')
        interpolated_recalls = np.linspace(0, 1, 100)
        interpolated_precisions = []

        for sorted_gnd in tqdm(all_sorted_gnds):
            precisions, recalls = [], []
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

    def compute_bit_entropy(self, hash_codes):
        binary_codes = (hash_codes > 0).astype(np.int32)  # shape: [N, K]
        entropies = []
        for bit in range(binary_codes.shape[1]):
            p = np.mean(binary_codes[:, bit])
            if p == 0 or p == 1:
                entropy = 0.0
            else:
                entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            entropies.append(entropy)
        return np.array(entropies)
