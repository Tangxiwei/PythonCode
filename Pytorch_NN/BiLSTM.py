import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.font_manager import FontProperties
# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 宋体字体路径
font_chinese = FontProperties(fname=font_path,size=18)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取数据
file_path = "D:/datasets/data952.xlsx"
datasets = pd.read_excel(file_path, header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]

# 数据截断和归一化
truncation = 0
data_list = datasets.iloc[truncation:].to_numpy()
scalers = [MinMaxScaler() for _ in datasets.columns]
data_scaled = np.column_stack(
    [scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

# 提取特征和目标信号
features = data_scaled[:, :-1]  # 所有列除了最后一列fairlead
target = data_scaled[:, -1]  # 最后一列fairlead

# 创建数据集
def create_dataset(features, target, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback, :])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)

lookback = 8
X, y = create_dataset(features, target, lookback)

# 数据集划分
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 定义BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.5):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                              bidirectional=True, dropout=dropout_prob)

        # 全连接层
        self.linear = nn.Linear(hidden_size * 2, output_size)  # hidden_size * 2 for bidirectional output

    def forward(self, x):
        # BiLSTM
        x, _ = self.bilstm(x)

        # 全连接层
        x = self.linear(x[:, -1, :])
        return x

# 实例化模型
input_size = X_train.shape[2]  # 特征数量
output_size = 1  # 预测一个目标值
model = BiLSTMModel(input_size=input_size, hidden_size=64, output_size=output_size, num_layers=2, dropout_prob=0.5).to(device)
# criterion = nn.MSELoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.0005)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
#
# # 早停机制
# patience = 10
# best_loss = np.inf
# epochs_no_improve = 0
#
# # 数据集和数据加载器
# train_dataset = TensorDataset(X_train, y_train)
# loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#
# # 模型训练
# n_epochs = 100
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in loader:
#         y_pred = model(X_batch).squeeze()
#         loss = criterion(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Validation
#     model.eval()
#     with torch.no_grad():
#         y_pred_test = model(X_test).squeeze()
#         test_loss = criterion(y_pred_test, y_test)
#         scheduler.step(test_loss)  # 更新学习率
#
#     # 早停机制
#     if test_loss < best_loss:
#         best_loss = test_loss
#         epochs_no_improve = 0
#         torch.save(model.state_dict(), 'best_model_weights_w25_BiLSTM.pth')
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve == patience:
#             print(f'Early stopping! No improvement for {patience} epochs.')
#             break
#
#     # 打印训练信息
#     if epoch % 5 == 0:
#         with torch.no_grad():
#             y_pred_train = model(X_train).squeeze()
#             train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
#             test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
#         print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")

# 加载最佳模型
model.load_state_dict(torch.load('best_model_weights_952_BiLSTM.pth'))

# 预测并反归一化
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
    y_test_original = data_list[:, -1][split_index+lookback:]
    y_pred_inv = y_pred_inv / 1000
    y_test_original = y_test_original / 1000

    # 计算评估指标
    test_rmse_value = np.sqrt(np.mean((y_pred_inv - y_test_original) ** 2))
    mae_value = np.mean(np.abs(y_pred_inv - y_test_original))
    r2_value = r2_score(y_test_original, y_pred_inv)
    test_mape_value = np.mean(np.abs((y_pred_inv - y_test_original) / y_test_original)) * 100
    print(test_rmse_value, r2_value, mae_value, test_mape_value)

# 绘制预测结果
file_path = "output952.xlsx"  # 这一部分是将数据输出
df = pd.read_excel(file_path)
df.insert(13, 'BiLSTM', y_pred_inv)  # 在第二列（索引为1的位置）插入数据
# # 将修改后的 DataFrame 写回 Excel 文件（覆盖原文件）
df.to_excel(file_path, index=False)  # index=False 表示不保存索引列

x = np.arange(0, len(y_pred_inv) * 0.05, 0.05) #0.2是我的时间间隔
x = x[1:]

print(test_rmse_value,r2_value,mae_value, test_mape_value)
# Plot results with enhanced aesthetics
plt.figure(figsize=(12, 10.5))
plt.ylim(0, 9000)
ticks = np.arange(0, 9100, 1500)  # 注意：arange 的结束值是开区间，所以要加 1
plt.yticks(ticks)
plt.tick_params(axis='both', which='major', labelsize=24)
# plt.plot(x_new,y_test_original[lookback:], label='Original Data', color='blue', linewidth=1)
# plt.plot(x_new,y_pred_inv, label='Predicted Data (CNN+LSTM+Attention)', color='red', linestyle='--', linewidth=1.8)
plt.plot(x,y_pred_inv, label='预测值 (BiLSTM)',color='#FF5733', linestyle='--', linewidth=2)
plt.plot(x,y_test_original, label='真实值', color='#2E86C1', linewidth=2)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('时间 (秒)', fontsize=24, fontproperties = font_chinese)
plt.ylabel('张力 (千牛)', fontsize=24, fontproperties = font_chinese)
plt.legend(  prop={
        'size': 24,          # 字体大小
        'family': 'SimSun',  # 字体族
    })
plt.text(0.55, 0.92, f'MAPE: {test_mape_value:.2f}%', transform=plt.gca().transAxes, fontsize=26,
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
# plt.savefig("cn_bilstmw25.png", dpi = 200 )
plt.show()