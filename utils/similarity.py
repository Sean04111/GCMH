import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm


def cosine_similarity(m1, m2):
    """
    计算两个矩阵的余弦相似度
    :param m1: 矩阵1
    :param m2: 矩阵2
    :return: 相似度
    """
    return m1.mm(m2.t())

def plot_sgc_distribution(sgc_matrix, save_path="sgc_distribution.png", show_progress=True):
    """
    绘制 Sgc 相似度矩阵的分布直方图（排除对角线）
    支持大矩阵绘图进度展示
    """

    if isinstance(sgc_matrix, torch.Tensor):
        sgc_matrix = sgc_matrix.detach().cpu().numpy()

    N = sgc_matrix.shape[0]
    values = []

    iterator = tqdm(range(N), desc="Extracting Sgc values") if show_progress else range(N)
    for i in iterator:
        for j in range(N):
            if i != j:
                values.append(sgc_matrix[i, j])

    values = np.array(values)

    # 绘图
    plt.figure(figsize=(8, 5))
    sns.histplot(values, bins=50, kde=True, color='skyblue')
    plt.title("Sgc Similarity Distribution (off-diagonal)")
    plt.xlabel("Similarity Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path