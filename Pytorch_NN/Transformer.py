import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
# from scipy.signal import savgol_filter
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 宋体字体路径
font_chinese = FontProperties(fname=font_path,size=18)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题'
# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# 读取数据
datasets = pd.read_excel(r"D:\datasets\data1267.xlsx", header=0)  # 读你自己的文件
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]

# 数据截断和缩放
truncation = 0
lookback = 6  # 需要时间依赖在这里设置
dims = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
data_list = datasets[dims].iloc[truncation:].to_numpy()

# 归一化数据
scalers = [MinMaxScaler() for _ in dims]
data_scaled = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

def create_dataset(data_scaled, lookback):
    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i + lookback, :-1])
        y.append(data_scaled[i + lookback, -1])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, lookback)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)
y_data = data_list[:, -1]
split_index = int(0.8 * len(X))
data_train, data_test = y_data[:split_index], y_data[split_index:]

# Transformer Model
class ShortTermTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, num_layers, local_window):
        super(ShortTermTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)
        self.layers = nn.ModuleList([
            LocalAttention(embed_size, num_heads, local_window) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, 1)  # 输出为单一值

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x[:, -1, :])  # 取最后一个时间步的输出

class LocalAttention(nn.Module):
    def __init__(self, embed_size, num_heads, local_window):
        super(LocalAttention, self).__init__()
        self.local_window = local_window
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.conv = nn.Conv1d(embed_size, embed_size, kernel_size=local_window, padding=local_window//2)

    def forward(self, x):
        # 使用卷积实现局部注意力
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        x = self.conv(x)  # 局部卷积
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, embed_size)
        # 使用多头注意力
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# 实例化Transformer模型
model = ShortTermTransformer(input_dim=6, embed_size=64, num_heads=2, num_layers=4, local_window=4).to(device)
# criterion = nn.MSELoss().to(device)
# # 数据加载器
# train_dataset = TensorDataset(X_train, y_train)
# # 修改学习率和批量大小
# optimizer = optim.Adam(model.parameters(), lr=0.0005)
# loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
# # 增加训练轮数
# n_epochs = 100
#
# # 使用学习率调度器
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
#
# # 早停
# best_loss = float('inf')
# patience = 10
# counter = 0
#
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in loader:
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # 验证
#     model.eval()
#     with torch.no_grad():
#         y_pred_train = model(X_train)
#         train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
#         y_pred_test = model(X_test)
#         test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
#
#     scheduler.step(test_rmse)
#
#     if test_rmse < best_loss:
#         best_loss = test_rmse
#         counter = 0
#     else:
#         counter += 1
#
#     if counter >= patience:
#         print(f"Early stopping at epoch {epoch}")
#         break
#
#     print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")
# #
# # # 保存模型权重
# torch.save(model.state_dict(), 'model_weights_952_transformer.pth')
model.load_state_dict(torch.load('model_weights_1267_transformer.pth'))

# 测试集预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.squeeze()
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()


y_test_original = data_list[:, -1][split_index:]

y_pred_inv = y_pred_inv/1000
y_test_original = y_test_original/1000
# 计算RMSE、MAE和R2
test_rmse_value = np.sqrt(np.mean((y_pred_inv - y_test_original[lookback:]) ** 2))
mae_value = np.mean(np.abs(y_pred_inv - y_test_original[lookback:]))
r2_value = r2_score(y_test_original[lookback:], y_pred_inv)
test_mape_value = np.mean(np.abs((y_pred_inv - y_test_original[lookback:]) / y_test_original[lookback:])) * 100

# file_path = "output1267.xlsx"  # 这一部分是将数据输出
# df = pd.read_excel(file_path)
# df.insert(4, 'Transformer', y_pred_inv)  # 在第二列（索引为1的位置）插入数据
# # # 将修改后的 DataFrame 写回 Excel 文件（覆盖原文件）
# df.to_excel(file_path, index=False)  # index=False 表示不保存索引列

# 打印评估指标
print(f"Test RMSE: {test_rmse_value:.4f}, R2: {r2_value:.4f}, MAE: {mae_value:.4f}, MAPE: {test_mape_value:.2f}%")

# 可视化结果
x = np.arange(0, len(y_pred_inv) * 0.2, 0.2)  # 0.2是我的时间间隔
x = x[1:]

# plt.figure(figsize=(12, 7))
# plt.plot(x, y_pred_inv, label='Predicted Data (LA-Transformer)', color='#FF5733', linestyle='--', linewidth=2)
# plt.plot(x, y_test_original[lookback:], label='Actual Value', color='#2E86C1', linewidth=2)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
# plt.ylim(700,2300)
# ticks = np.arange(700,2301,200)  # 注意：arange 的结束值是开区间，所以要加 1
# plt.yticks(ticks)
# plt.xlabel('Time (s)', fontsize=18)
# plt.ylabel('Tension (kN)', fontsize=18)
# plt.legend(fontsize=16, loc='upper left')
# plt.text(0.55, 0.94, f'Test MAPE: {test_mape_value:.2f}%', transform=plt.gca().transAxes, fontsize=18,
#          bbox=dict(facecolor='white', alpha=0.8))
# # plt.savefig("trans75.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
# plt.tight_layout()
# plt.show()

x = np.arange(0, len(y_pred_inv) * 0.05, 0.05)  # 0.2 is the time interval
x = x[1:]
plt.figure(figsize=(12, 7))
plt.ylim(0,10000)
ticks = np.arange(0,10001, 2000)  # 注意：arange 的结束值是开区间，所以要加 1
plt.yticks(ticks)
plt.tick_params(axis='both', which='major', labelsize=20)
# plt.plot(x_new,y_test_original[lookback:], label='Original Data', color='blue', linewidth=1)
# plt.plot(x_new,y_pred_inv, label='Predicted Data (CNN+LSTM+Attention)', color='red', linestyle='--', linewidth=1.8)
plt.plot(x,y_pred_inv, label='预测值 (ST-Transformer)',color='#FF5733', linestyle='--', linewidth=2)
plt.plot(x,y_test_original[lookback:], label='真实值', color='#2E86C1', linewidth=2)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('时间 (秒)', fontsize=20, fontproperties = font_chinese)
plt.ylabel('张力 (千牛)', fontsize=20, fontproperties = font_chinese)
plt.legend(  prop={
        'size': 20,          # 字体大小
        'family': 'SimSun',  # 字体族
    })
plt.text(0.55, 0.92, f'MAPE: {test_mape_value:.2f}%', transform=plt.gca().transAxes, fontsize=26,
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig("cn_Transformer_1267.png", dpi = 200 )
plt.show()
