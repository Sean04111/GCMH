import torch
import torch.nn.functional as F

# ranking loss 的margin退火

def get_margin(epoch, total_epoch, max_margin = 0.5, min_margin = 0.1):
    progress = epoch / total_epoch
    return max_margin * (1 - progress) + min_margin * progress

def ranking_loss(HI, HT, Sgc, margin, pos_thresh=0.6, neg_thresh=0.3):
    """
    Faster Cross-modal Ranking Loss using vectorized computation and CUDA support.
    """

    HI = F.normalize(HI, dim=1)
    HT = F.normalize(HT, dim=1)
    sim_i2t = torch.matmul(HI, HT.T)  # [B, B]
    sim_t2i = sim_i2t.T               # [B, B]
    B = HI.size(0)

    device = HI.device

    # 构造正负样本掩码
    pos_mask = Sgc > pos_thresh  # [B, B]
    neg_mask = Sgc < neg_thresh  # [B, B]

    # -------- I → T 方向 --------
    i_idx, pos_j = pos_mask.nonzero(as_tuple=True)
    _, neg_k = neg_mask.nonzero(as_tuple=True)

    # 构造三元组索引 i-j-k，确保样本 i 的负样本来自于 i 行
    i_expand = i_idx.repeat_interleave(len(neg_k))
    pos_j_expand = pos_j.repeat_interleave(len(neg_k))
    neg_k_expand = neg_k.repeat(len(i_idx))

    valid_mask = (i_expand < B) & (pos_j_expand < B) & (neg_k_expand < B)
    i_expand = i_expand[valid_mask]
    pos_j_expand = pos_j_expand[valid_mask]
    neg_k_expand = neg_k_expand[valid_mask]

    sim_pos = sim_i2t[i_expand, pos_j_expand]
    sim_neg = sim_i2t[i_expand, neg_k_expand]
    loss_i2t = F.relu(sim_neg - sim_pos + margin)

    # -------- T → I 方向 --------
    j_idx, pos_i = pos_mask.T.nonzero(as_tuple=True)
    _, neg_k = neg_mask.T.nonzero(as_tuple=True)

    j_expand = j_idx.repeat_interleave(len(neg_k))
    pos_i_expand = pos_i.repeat_interleave(len(neg_k))
    neg_k_expand = neg_k.repeat(len(j_idx))

    valid_mask = (j_expand < B) & (pos_i_expand < B) & (neg_k_expand < B)
    j_expand = j_expand[valid_mask]
    pos_i_expand = pos_i_expand[valid_mask]
    neg_k_expand = neg_k_expand[valid_mask]

    sim_pos = sim_t2i[j_expand, pos_i_expand]
    sim_neg = sim_t2i[j_expand, neg_k_expand]
    loss_t2i = F.relu(sim_neg - sim_pos + margin)

    total_loss = torch.cat([loss_i2t, loss_t2i], dim=0)
    if total_loss.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    return total_loss.mean()


def entropy_loss(H):
    """
    H: Tensor of shape [B, D] in range [-1, 1]
    作用：鼓励每一位 bit 在整个 batch 中更均匀（信息熵更高）
    """
    H = (H + 1) / 2  # [-1,1] → [0,1]
    p = torch.mean(H, dim=0)  # [D] 每一位的正值概率
    entropy = -p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
    return -torch.mean(entropy)  # 负号使其作为 loss（最大化熵）



