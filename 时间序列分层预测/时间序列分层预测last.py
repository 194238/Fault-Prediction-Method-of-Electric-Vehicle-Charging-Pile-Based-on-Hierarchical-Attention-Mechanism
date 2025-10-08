import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hierarchy_data import ChargingHierarchyData
from utils import lag_dataset
import random
import os

# -------------------------------
# 1. 超参数与随机种子设置
# -------------------------------
SEED = 42
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
AHEAD = 12  # 预测步长
TRAIN_UPTO = 600  # 训练、验证数据的时间步数  732 -> 12
BACKUP_TIME = 24  # 构建样本时用到的历史时间步数
BATCH_SIZE = 24
TRAIN_LR = 0.001
LAMBDA = 0.5  # 一致性正则化损失权重
TRAIN_EPOCHS = 1  # 训练轮数
C = 5.0  # sigma 修正中的超参数

# 学习率调度器参数
LR_STEP_SIZE = 10   # 每5个epoch调整一次学习率
LR_GAMMA = 0.5     # 学习率调整的乘数因子

np.random.seed(SEED)
th.manual_seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# -------------------------------
# 2. 数据加载与预处理
# -------------------------------
data_obj = ChargingHierarchyData()
full_data = data_obj.data  # 假设 shape 为 [num_nodes, total_timesteps]
# 取训练数据（每个节点前 TRAIN_UPTO 个时间步）
train_data_raw = full_data[:, :TRAIN_UPTO]
# 划分： 75% 训练，25% 验证
train_len = int(TRAIN_UPTO * 0.75)  #  450 个时间步
val_len = TRAIN_UPTO - train_len   #  150 个时间步

actual_train_data_raw = full_data[:, :train_len]  # 训练集
val_data_raw = full_data[:, train_len:TRAIN_UPTO]   # 验证集

# 按节点归一化（使用训练集的统计量）
train_means = np.mean(actual_train_data_raw, axis=1)
train_std = np.std(actual_train_data_raw, axis=1)

train_data = (actual_train_data_raw - train_means[:, None]) / (train_std[:, None] + 1e-6)

# 验证集归一化：使用训练集的均值和标准差
val_data = (val_data_raw - train_means[:, None]) / (train_std[:, None] + 1e-6)

# 测试集归一化可以保持原来的处理方式，如果希望用测试集自己的统计量，可以单独计算
test_data_raw = full_data[:, TRAIN_UPTO:]
# test_means = np.mean(test_data_raw, axis=1)
# test_std = np.std(test_data_raw, axis=1)
# 此处仍用训练集统计量以保持一致性
test_data = (test_data_raw - train_means[:, None]) / (train_std[:, None] + 1e-6)


# 反归一化
def inverse_transform(pred, means, stds):
    """
    反归一化预测值，使其恢复到原始尺度。
    :param pred: 预测值张量 (num_samples, num_nodes, ahead)
    :param means: 训练数据均值 (num_nodes,)
    :param stds: 训练数据标准差 (num_nodes,)
    :return: 反归一化后的预测值 (num_samples, num_nodes, ahead)
    """
    return pred * stds[:, None] + means[:, None]  # 确保广播匹配


# 利用 lag_dataset 构造训练样本
# 假设 lag_dataset 返回 (X, Y)：
# X: [num_samples, num_nodes, BACKUP_TIME]，Y: [num_samples, num_nodes, AHEAD]
train_dataset_raw = lag_dataset(train_data, BACKUP_TIME, ahead=AHEAD)
val_dataset_raw = lag_dataset(val_data, BACKUP_TIME, ahead=AHEAD)
test_dataset_raw = lag_dataset(test_data, BACKUP_TIME, ahead=AHEAD)

# 根据 ChargingHierarchyData 中构建好的节点（data_obj.nodes）构造层级关系字典
hierarchy_info = {}
for node in data_obj.nodes:
    if node.children:  # 如果该节点有子节点
        # 收集所有子节点的索引（应该使用 idx 而不是 id）
        children_ids = [child.idx for child in node.children]

        # 计算均匀分布的聚合系数 phi（每个子节点的权重相等，和为 1）
        # phi_values = th.tensor([1.0 / len(children_ids)] * len(children_ids))   # 所有节点的均值
        phi_values = th.tensor(1.0)    # 所有节点的求和

        # 存入层级关系字典
        hierarchy_info[node.idx] = {'children': children_ids, 'phi': phi_values}
    else:
        # 叶子节点无子节点，phi 设为空张量
        hierarchy_info[node.idx] = {'children': [], 'phi': th.tensor([])}


# -------------------------------
# 3. 定义数据集包装类
# -------------------------------
class SeqDataset(Dataset):
    def __init__(self, dataset):
        self.X, self.Y = dataset
        # X: [num_samples, num_nodes, BACKUP_TIME], Y: [num_samples, num_nodes, AHEAD]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# 创建数据集包装类
train_dataset = SeqDataset(train_dataset_raw)
val_dataset = SeqDataset(val_dataset_raw)
test_dataset = SeqDataset(test_dataset_raw)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 验证集一般不打乱
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------------
# 4. 构造邻接矩阵（邻域）——用于新模型的层次注意力模块
# -------------------------------
def build_neighbor_mask(hierarchy_info, num_nodes):
    # 构造一个 [num_nodes, num_nodes] 的邻接矩阵，若两个节点具有直接层次关系（父子或相互）则为1，并包含自连接
    mask = th.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        mask[i, i] = 1  # 自连接
    for parent, info in hierarchy_info.items():
        for child in info['children']:
            mask[parent, child] = 1
            mask[child, parent] = 1  # 双向连接
    return mask


# 在数据预处理阶段构建 parent_child_mask
def build_parent_child_mask(hierarchy_info, num_nodes):
    """
    构建父节点到子节点的单向关系矩阵
    :param hierarchy_info: 层级关系字典，包含每个节点的子节点列表
    :return: [num_nodes, num_nodes] 的矩阵，父节点到子节点为1，其他为0
    """
    mask = th.zeros(num_nodes, num_nodes)
    for parent, info in hierarchy_info.items():
        for child in info['children']:
            mask[parent, child] = 1  # 单向：父关注子
    return mask


num_nodes = full_data.shape[0]
neighbor_mask = build_neighbor_mask(hierarchy_info, num_nodes).to(DEVICE)  # [num_nodes, num_nodes]
# 构造层级关系矩阵
parent_child_mask = build_parent_child_mask(hierarchy_info, num_nodes).to(DEVICE)

# -------------------------------
# 5. 模型定义
# 5.1 基础预测模型（对每个节点的历史序列进行预测）
# -------------------------------
class BaseForecastAggregated(nn.Module):
    def __init__(self, history_len, ahead, d_model=64, nhead=4, num_layers=2, dropout_rate=0.2):
        super(BaseForecastAggregated, self).__init__()
        self.input_projection = nn.Linear(1, d_model)  # 线性变换，将 1 维输入扩展到 d_model
        self.positional_encoding = nn.Parameter(th.zeros(1, history_len, d_model))  # 可训练的位置编码
        # 基础 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 预测输出
        self.fc_mu = nn.Linear(d_model, ahead)
        self.fc_log_sigma = nn.Linear(d_model, ahead)

    def forward(self, x):
        # x: [B, history_len]
        x = x.unsqueeze(-1)  # 变成 [B, history_len, 1]
        x = self.input_projection(x)  # 变成 [B, history_len, d_model]
        x = x + self.positional_encoding  # 位置编码
        x = self.transformer(x)  # Transformer 处理，输出 [B, history_len, d_model]
        x = x[:, -1, :]  # 取最后一个时间步的表示 [B, d_model]
        mu = self.fc_mu(x)  # 预测值
        log_sigma = self.fc_log_sigma(x)  # 预测标准差
        sigma = th.exp(log_sigma) + 1e-6  # 避免数值问题
        return mu, sigma


class BaseForecastPile(nn.Module):
    def __init__(self, history_len, ahead, hidden_dim=64, lstm_hidden=32, num_layers=2, dropout_rate=0.2):
        super(BaseForecastPile, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_rate
        )
        # 由于是双向 LSTM，输出特征维度为 2*lstm_hidden
        self.fc1 = nn.Linear(2 * lstm_hidden, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(hidden_dim, ahead)
        self.fc_log_sigma = nn.Linear(hidden_dim, ahead)
        # 可选：残差连接，将双向 LSTM 的输出直接映射到 hidden_dim
        self.residual = nn.Linear(2 * lstm_hidden, hidden_dim)

    def forward(self, x):
        # x: [B, history_len] -> 扩展为 [B, history_len, 1]
        x = x.unsqueeze(-1)
        lstm_out, (h_n, _) = self.lstm(x)
        # 取最后一个时刻的双向输出：拼接正向与反向输出
        # 如果使用多层 LSTM，可以考虑拼接所有层的最后时刻输出
        last_forward = h_n[-2, :, :]  # [B, lstm_hidden]
        last_backward = h_n[-1, :, :]  # [B, lstm_hidden]
        last_hidden = th.cat([last_forward, last_backward], dim=-1)  # [B, 2*lstm_hidden]
        # 加入残差连接
        residual = self.residual(last_hidden)
        h = F.gelu(self.bn1(self.fc1(last_hidden)))
        h = self.dropout1(h)
        h = h + residual  # 残差连接
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        sigma = th.exp(log_sigma) + 1e-6
        return mu, sigma


# 多节点并行预测，即MLP同时对167个节点进行预测，LSTM同时对剩下的所有节点进行预测
class BaseForecastAggregatedWrapper(nn.Module):
    def __init__(self, num_agg, history_len, ahead, hidden_dim=64):
        super(BaseForecastAggregatedWrapper, self).__init__()
        self.num_agg = num_agg
        self.module = BaseForecastAggregated(history_len, ahead, hidden_dim)

    def forward(self, x):
        # x: [B, num_agg, history_len]
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        mu, sigma = self.module(x)
        mu = mu.view(B, N, -1)
        sigma = sigma.view(B, N, -1)
        return mu, sigma


class BaseForecastPileWrapper(nn.Module):
    def __init__(self, num_pile, history_len, ahead, hidden_dim=64, lstm_hidden=32):
        super(BaseForecastPileWrapper, self).__init__()
        self.num_pile = num_pile
        self.module = BaseForecastPile(history_len, ahead, hidden_dim, lstm_hidden)

    def forward(self, x):
        # x: [B, num_pile, history_len]
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        mu, sigma = self.module(x)
        mu = mu.view(B, N, -1)
        sigma = sigma.view(B, N, -1)
        return mu, sigma


class DualBaseForecastModel(nn.Module):
    def __init__(self, num_total, num_agg, history_len, ahead, hidden_dim=64, lstm_hidden=32):
        super(DualBaseForecastModel, self).__init__()
        num_pile = num_total - num_agg
        self.num_agg = num_agg
        self.agg_model = BaseForecastAggregatedWrapper(num_agg, history_len, ahead, hidden_dim)
        self.pile_model = BaseForecastPileWrapper(num_pile, history_len, ahead, hidden_dim, lstm_hidden)

    def forward(self, x):
        # x: [B, num_total, history_len]
        x_agg = x[:, :self.num_agg, :]  # 前 167 个节点
        x_pile = x[:, self.num_agg:, :]  # 剩余节点
        mu_agg, sigma_agg = self.agg_model(x_agg)
        mu_pile, sigma_pile = self.pile_model(x_pile)
        mu = th.cat([mu_agg, mu_pile], dim=1)
        sigma = th.cat([sigma_agg, sigma_pile], dim=1)
        return mu, sigma


# -------------------------------
# 5.2 层次注意力精炼模块
# 采用自注意力机制对每个节点的基础预测特征进行信息融合，
# 仅在“邻域”（由邻接矩阵 neighbor_mask 指定）内进行注意力聚合。
class HierAttnRefine(nn.Module):
    def __init__(self, num_nodes, input_dim, attn_hidden_dim):
        super(HierAttnRefine, self).__init__()
        self.query = nn.Linear(input_dim, attn_hidden_dim)
        self.key = nn.Linear(input_dim, attn_hidden_dim)
        self.value = nn.Linear(input_dim, attn_hidden_dim)
        self.out = nn.Linear(attn_hidden_dim, input_dim)

        # 可学习参数
        self.alpha = nn.Parameter(th.tensor(2.0))  # 注意力偏置强度
        self.gamma = nn.Parameter(th.tensor(0.3))  # 残差修正强度

    def forward(self, base_features, neighbor_mask, parent_child_mask=None):
        B, N, D = base_features.shape
        Q, K, V = self.query(base_features), self.key(base_features), self.value(base_features)

        # 计算注意力得分
        scores = th.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))

        # 层级偏置注入
        if parent_child_mask is not None:
            hierarchy_bias = parent_child_mask.unsqueeze(0) * self.alpha
            scores = scores + hierarchy_bias

        # 掩码处理
        scores = scores.masked_fill(neighbor_mask.unsqueeze(0) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # 特征聚合
        attn_output = th.matmul(attn_weights, V)
        refined_features = self.out(attn_output) + base_features

        # 层级残差修正
        if parent_child_mask is not None:
            child_sum = th.matmul(parent_child_mask.float(), refined_features)
            is_parent = parent_child_mask.sum(dim=1) > 0  # [N]
            delta = self.gamma * (child_sum - refined_features)
            refined_features = refined_features + delta * is_parent[None, :, None]

        return refined_features
# -------------------------------
# class HierAttnRefine(nn.Module):
#     def __init__(self, num_nodes, input_dim, attn_hidden_dim):
#         super(HierAttnRefine, self).__init__()
#         # 定义 query, key, value 映射
#         self.query = nn.Linear(input_dim, attn_hidden_dim)
#         self.key = nn.Linear(input_dim, attn_hidden_dim)
#         self.value = nn.Linear(input_dim, attn_hidden_dim)
#         self.out = nn.Linear(attn_hidden_dim, input_dim)  # 将注意力输出映射回原特征维度
#
#     def forward(self, base_features, neighbor_mask):
#         # base_features: [B, num_nodes, input_dim]
#         # neighbor_mask: [num_nodes, num_nodes]，取值0或1，若 (i,j) 为邻居则为1
#         B, N, D = base_features.shape
#         Q = self.query(base_features)  # [B, N, attn_hidden_dim]
#         K = self.key(base_features)  # [B, N, attn_hidden_dim]
#         V = self.value(base_features)  # [B, N, attn_hidden_dim]
#         # 计算注意力得分
#         scores = th.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))  # [B, N, N]
#         # 将非邻居位置的得分置为极小值
#         mask = neighbor_mask.unsqueeze(0)  # [1, N, N]
#         scores = scores.masked_fill(mask == 0, -1e9)
#         attn_weights = F.softmax(scores, dim=-1)  # [B, N, N]
#         attn_output = th.matmul(attn_weights, V)  # [B, N, attn_hidden_dim]
#         refined_features = self.out(attn_output) + base_features  # 残差连接
#         return refined_features


# -------------------------------
# 5.3 新模型：HierAttnForecast
# 该模型由基础预测模块和层次注意力精炼模块组成，
# 其中先用基础模型得到每个节点的预测均值（作为特征），
# 经由线性变换映射到较低维度，再通过注意力模块融合邻域信息，
# 最后映射回预测空间输出 refined_mu 和 refined_sigma。
# -------------------------------
class HierAttnForecastDual(nn.Module):
    def __init__(self, num_total, num_agg, history_len, ahead, base_hidden_dim=64, attn_hidden_dim=32, lstm_hidden=32):
        super(HierAttnForecastDual, self).__init__()
        self.num_total = num_total
        self.ahead = ahead
        # 使用双基础预测模块
        self.dual_base_model = DualBaseForecastModel(num_total, num_agg, history_len, ahead, base_hidden_dim, lstm_hidden)
        # 将基础预测均值投影到较低维度作为注意力输入
        self.feature_proj = nn.Linear(ahead, attn_hidden_dim)
        # 层次注意力精炼模块（保持不变）
        self.hier_attn = HierAttnRefine(num_total, attn_hidden_dim, attn_hidden_dim)
        # 输出层
        self.out_mu = nn.Linear(attn_hidden_dim, ahead)
        self.out_log_sigma = nn.Linear(attn_hidden_dim, ahead)

    def forward(self, x, neighbor_mask):
        base_mu, base_sigma = self.dual_base_model(x)
        features = self.feature_proj(base_mu)
        refined_features = self.hier_attn(features, neighbor_mask)
        refined_mu = self.out_mu(refined_features)
        refined_log_sigma = self.out_log_sigma(refined_features)
        refined_sigma = th.exp(refined_log_sigma) + 1e-6
        return refined_mu, refined_sigma


# -------------------------------
# 6. 损失函数定义
# 6.1 高斯负对数似然损失
# -------------------------------
def gaussian_nll_loss(y_true, mu, sigma):
    loss = 0.5 * th.log(2 * math.pi * sigma ** 2) + 0.5 * ((y_true - mu) ** 2) / (sigma ** 2)
    return loss.mean()


# -------------------------------
# 6.2 层次一致性正则化（可选）
# 对于每个非叶父节点，要求其 refined 预测分布与子节点加权聚合预测分布之间的一致性
# -------------------------------
def soft_consistency_loss(refined_mu, refined_sigma, hierarchy_info, eps=1e-6):
    loss_total = 0.0
    count = 0
    B, N, T = refined_mu.shape
    for parent, info in hierarchy_info.items():
        children = info['children']
        if len(children) == 0:
            continue
        phi = info['phi'].to(refined_mu.device).view(1, -1, 1)
        parent_mu = refined_mu[:, parent, :]  # [B, ahead]
        parent_sigma_sq = refined_sigma[:, parent, :] ** 2
        children_mu = refined_mu[:, children, :]  # [B, num_children, ahead]
        children_sigma_sq = refined_sigma[:, children, :] ** 2
        agg_children_mu = th.sum(phi * children_mu, dim=1)  # [B, ahead]
        agg_children_sigma_sq = th.sum((phi ** 2) * children_sigma_sq, dim=1)
        diff_sq = (parent_mu - agg_children_mu) ** 2
        term1 = parent_sigma_sq
        term2 = diff_sq / (4.0 * (agg_children_sigma_sq + eps))
        term3 = agg_children_sigma_sq
        term4 = diff_sq / (4.0 * (parent_sigma_sq + eps))
        loss_i = term1 + term2 + term3 + term4 - 0.5
        loss_total += loss_i.mean()
        count += 1
    return loss_total / count if count > 0 else th.tensor(0.0, device=refined_mu.device)


# -------------------------------
# 7. 训练与评估流程
# -------------------------------
def train_model(model, dataloader, optimizer, hierarchy_info, neighbor_mask, lambda_consistency, num_epochs, scheduler=None):
    model.train()
    overall_losses = []
    ll_losses = []
    cons_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_loss_ll = 0.0
        total_loss_cons = 0.0
        num_batches = len(dataloader)
        for x_batch, y_batch in dataloader:
            # x_batch: [B, num_nodes, BACKUP_TIME]，y_batch: [B, num_nodes, AHEAD]
            x_batch = x_batch.float().to(DEVICE)
            y_batch = y_batch.float().to(DEVICE)
            optimizer.zero_grad()
            refined_mu, refined_sigma = model(x_batch, neighbor_mask)
            loss_ll = gaussian_nll_loss(y_batch, refined_mu, refined_sigma)
            loss_cons = soft_consistency_loss(refined_mu, refined_sigma, hierarchy_info)
            loss = loss_ll + lambda_consistency * loss_cons
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_ll += loss_ll.item()
            total_loss_cons += loss_cons.item()

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        epoch_loss = total_loss / num_batches
        epoch_loss_ll = total_loss_ll / num_batches
        epoch_loss_cons = total_loss_cons / num_batches

        overall_losses.append(epoch_loss)
        ll_losses.append(epoch_loss_ll)
        cons_losses.append(epoch_loss_cons)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Loss_ll: {epoch_loss_ll:.4f}, Loss_cons: {epoch_loss_cons:.4f}")

    # 保存损失函数的记录到 CSV 文件
    import pandas as pd
    epochs = list(range(1, num_epochs + 1))

    df_overall = pd.DataFrame({"epoch": epochs, "overall_loss": overall_losses})
    df_ll = pd.DataFrame({"epoch": epochs, "loss_ll": ll_losses})
    df_cons = pd.DataFrame({"epoch": epochs, "loss_cons": cons_losses})

    df_overall.to_csv("overall_loss.csv", index=False)
    df_ll.to_csv("loss_ll.csv", index=False)
    df_cons.to_csv("loss_cons.csv", index=False)



# def evaluate_model(model, dataloader, neighbor_mask, train_means, train_std, device=DEVICE):
#     model.eval()
#     all_mu = []
#     all_y = []
#     with th.no_grad():
#         for x_batch, y_batch in dataloader:
#             x_batch = x_batch.float().to(device)
#             y_batch = y_batch.float().to(device)
#             refined_mu, _ = model(x_batch, neighbor_mask)  # 取均值作为预测值
#             all_mu.append(refined_mu.cpu().numpy())
#             all_y.append(y_batch.cpu().numpy())
#
#     # 合并所有批次数据
#     pred = np.concatenate(all_mu, axis=0)  # [num_samples, num_nodes, ahead]
#     true = np.concatenate(all_y, axis=0)
#
#     # 反归一化到原始尺度 (使用训练集的统计量)
#     pred = inverse_transform(pred, train_means, train_std)
#     true = inverse_transform(true, train_means, train_std)
#
#     # ** 添加取整的约束条件 **
#     # 对预测的均值进行四舍五入，转换为整数
#     pred = np.round(pred).astype(int)
#     true = np.round(true).astype(int)
#
#     # 计算指标
#     mse = np.mean((pred - true) ** 2)
#     rmse = np.sqrt(mse)
#     mae = np.mean(np.abs(pred - true))
#
#     return {"MSE": mse, "RMSE": rmse, "MAE": mae}


# -------------------------------
# 8. 主流程：构造模型、训练、评估并保存预测结果与真实值
# -------------------------------
if __name__ == "__main__":
    # 根据 full_data 得到节点数
    num_nodes = full_data.shape[0]

    # import pdb
    # pdb.set_trace()

    # 构造新模型：HierAttnForecast
    model = HierAttnForecastDual(num_total=num_nodes, num_agg=167, history_len=BACKUP_TIME, ahead=AHEAD,
                                 base_hidden_dim=64, attn_hidden_dim=32).to(DEVICE)
    # optimizer = th.optim.Adam(model.parameters(), lr=TRAIN_LR)
    # 定义优化器和学习率调度器
    optimizer = th.optim.Adam(model.parameters(), lr=TRAIN_LR)
    scheduler = th.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA
    )
    print("Start training HierAttnForecast...")
    train_model(model, train_loader, optimizer, hierarchy_info, neighbor_mask,
                    lambda_consistency=LAMBDA, num_epochs=TRAIN_EPOCHS, scheduler=scheduler)

    # print("Start evaluate HierAttnForecast...")
    # evaluate_model(model, val_loader, neighbor_mask, train_means, train_std, DEVICE)

    # -------------------------------
    # 保存预测结果与真实值到 CSV 文件
    # -------------------------------
    import pandas as pd
    import torch as th


    def save_predictions_to_csv(model, dataloader, neighbor_mask, train_means, train_std,
                                pred_file="predictions.csv", true_file="true_values.csv"):
        model.eval()
        all_mus = []
        all_ys = []

        with th.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.float().to(DEVICE)
                y_batch = y_batch.float().to(DEVICE)
                refined_mu, _ = model(x_batch, neighbor_mask)
                all_mus.append(refined_mu.cpu().numpy())
                all_ys.append(y_batch.cpu().numpy())

        # 拼接所有批次数据
        all_mus = np.concatenate(all_mus, axis=0)  # 形状: (num_samples, num_nodes, ahead)
        all_ys = np.concatenate(all_ys, axis=0)

        num_samples, num_nodes, ahead = all_mus.shape

        # 生成元数据
        sample_ids = np.repeat(np.arange(num_samples), num_nodes)  # 样本ID重复num_nodes次
        node_ids = np.tile(np.arange(num_nodes), num_samples)  # 节点ID平铺num_samples次

        # 反归一化
        means = train_means[node_ids].reshape(-1, 1)
        stds = train_std[node_ids].reshape(-1, 1)

        # 重塑数据为 (num_samples*num_nodes, ahead)
        pred_values = (all_mus.reshape(-1, ahead) * stds) + means
        true_values = (all_ys.reshape(-1, ahead) * stds) + means

        # 创建列名（支持任意预测步长）
        pred_columns = [f"t+{i + 1}" for i in range(ahead)]

        # 构建预测DataFrame
        df_pred = pd.DataFrame(
            pred_values,
            columns=pred_columns
        )
        df_pred.insert(0, "sample_id", sample_ids)
        df_pred.insert(1, "node_id", node_ids)

        # 构建真实值DataFrame
        df_true = pd.DataFrame(
            true_values,
            columns=pred_columns  # 使用相同的列名
        )
        df_true.insert(0, "sample_id", sample_ids)
        df_true.insert(1, "node_id", node_ids)

        # 保存文件
        df_pred.to_csv(pred_file, index=False)
        df_true.to_csv(true_file, index=False)

        print(f"✅ Predictions saved to: {pred_file}")
        print(f"✅ True values saved to: {true_file}")


    # 调用示例
    save_predictions_to_csv(
        model,
        test_loader,
        neighbor_mask,
        train_means,  # 形状应为 (num_nodes,)
        train_std,  # 形状应为 (num_nodes,)
        pred_file="full_predictions.csv",
        true_file="full_true_values.csv"
    )
