
def cosine_similarity(m1, m2):
    """
    计算两个矩阵的余弦相似度
    :param m1: 矩阵1
    :param m2: 矩阵2
    :return: 相似度
    """
    return m1.mm(m2.t())