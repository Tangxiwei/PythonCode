import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import Polynomial
# 设置设备为GPU（如果可用）
device = torch.device('cuda')
print(torch.cuda.is_available())
datasets = pd.read_excel("D:/datasets/data0545.xlsx", header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
truncation = 39000
lookback = 6
dims = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
data_list = datasets[dims].iloc[truncation:].to_numpy()

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
y_data = data_list[:,6]
split_index = int(0.8 * len(y_data))
data_train, data_test = y_data[:split_index], y_data[split_index:]

pitch_train = data_list[:split_index, 4] * np.pi / 180
yaw_train = data_list[:split_index, 5] * np.pi / 180
pitch_test = data_list[split_index:,4]* np.pi/ 180
yaw_test = data_list[split_index:, 5] * np.pi / 180
xi_train = data_list[:split_index, 0] + 40.868 * (2 - np.cos(pitch_train) - np.cos(yaw_train))
xi_test = data_list[split_index:, 0] + 40.868 * (2 - np.cos(pitch_test) - np.cos(yaw_test))
# 一般多项式拟合
degrees = [6, 8, 10, 12, 14]
fits = []
mse_list = []
r2_list = []
for degree in degrees:
    coefs = Polynomial.fit(xi_train, data_train, degree)
    fits.append(coefs)
    y_fit = coefs(xi_train)
    mse = mean_squared_error(data_train, y_fit)
    r2 = r2_score(data_train, y_fit)
    mse_list.append(mse)
    r2_list.append(r2)
    print(f"Degree {degree}: MSE = {mse}, R² = {r2}")

best_degree = degrees[np.argmin(mse_list)]  # 或 degrees[np.argmax(r2_list)]
best_fit = fits[np.argmin(mse_list)]  # 或 fits[np.argmax(r2_list)]
print(f"Best degree: {best_degree}")
print(f"Coefficients: {best_fit.convert().coef}")
# 使用最优拟合模型对测试数据进行预测
y_test_fit = best_fit(xi_test)

class LocalAttentionTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, window_size=10):
        super(LocalAttentionTCN, self).__init__()
        self.window_size = window_size
        self.tcn = nn.Conv1d(input_size, num_channels, kernel_size, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=2)
        self.fc = nn.Linear(num_channels, output_size)
    def forward(self, x):
        # 只取最后 window_size 个时间步
        x_window = x[:, -self.window_size:, :]
        # 将输入转置为 (batch, channels, seq_len) 以适应 Conv1d
        x_window = x_window.permute(0, 2, 1)
        # TCN 卷积前向传播
        tcn_out = self.tcn(x_window)
        # 转置回来以适应多头注意力机制 (batch, seq_len, channels)
        tcn_out = tcn_out.permute(0, 2, 1)
        # 注意力机制
        attn_output, _ = self.attention(tcn_out, tcn_out, tcn_out)
        # 取最后一个时间步的注意力输出
        # out = attn_output[:, -1, :]
        out = torch.mean(attn_output, dim=1)
        # 输出层
        out = self.fc(out)
        return out
# 示例使用
# model = LocalAttentionTCN(input_size=6, output_size=1, num_channels=32, kernel_size=2, window_size=6).to(device)
# criterion = nn.MSELoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.002)
# train_dataset = TensorDataset(X_train, y_train)
# loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# n_epochs = 25
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in loader:
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         torch.cuda.empty_cache()
#         # Validation
#     if epoch % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             y_pred_train = model(X_train)
#             train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
#             y_pred_test = model(X_test)
#             test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
#         print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")
# torch.save(model.state_dict(), 'model_weights_0545_LATCN-ploy.pth')
model = LocalAttentionTCN(input_size=6, output_size=1, num_channels=32, kernel_size=2, window_size=6).to(device)
model.load_state_dict(torch.load('model_weights_0545_LATCN-ploy.pth'))
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
    y_test_original = data_list[:, -1][-len(y_test):]
lower_bound = 0.95*y_test_fit
upper_bound = 1.05*y_test_fit
y_pred_inv_poly = np.clip(y_pred_inv,lower_bound[2:],upper_bound[2:])
# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_test_fit[::4], label = 'Poly Fit',color = 'black')
plt.plot(y_pred_inv_poly[::4], label = "Ploy+LSTM", color = 'green')
plt.plot(y_test_original[::4], label='Original Data', color='blue')
plt.plot(y_pred_inv[::4], label='Predicted Data(LSTM)', color='red')
plt.legend()
plt.show()