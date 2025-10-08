import numpy as np

# def lag_dataset(seqs, back):
#     X, Y = [], []
#     for i in range(back, seqs.shape[1]):
#         X.append(seqs[:, :i-1])
#         Y.append(seqs[:, i])
#     return X, Y

# def lag_dataset(seqs, back):
#     X, Y = [], []
#     for i in range(back, seqs.shape[1]):
#         X.append(seqs[:, i-back:i])  # 取过去 `back` 步作为 X
#         Y.append(seqs[:, i])         # 取当前时间步作为 Y
#     return np.array(X), np.array(Y)  # 转为 NumPy 数组，确保形状一致


def lag_dataset(seqs, back, ahead):
    X, Y = [], []
    for i in range(back, seqs.shape[1] - ahead + 1):
        X.append(seqs[:, i-back:i])  # 过去 `back` 步
        Y.append(seqs[:, i:i+ahead])  # 未来 `ahead` 步
    return np.array(X), np.array(Y)  # 转为 NumPy 数组
