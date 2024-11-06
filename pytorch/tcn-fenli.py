from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

datasets = pd.read_excel("D:/datasets/data.1/data0545_30.xlsx", header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
truncation = 0
lookback = 8
dims = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
data_list = datasets[dims].iloc[truncation:].to_numpy()
# Normalize data
scalers = [MinMaxScaler() for _ in dims]
data_scaled = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

# 定义低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 高、低频分离函数
def split_high_low_frequencies(data, highcut, fs):
    low_freq_data = butter_lowpass_filter(data, highcut, fs)
    high_freq_data = data - low_freq_data
    return low_freq_data, high_freq_data
# 分离 Surge 和 Sway 信号
fs = 20.0  # 采样频率
highcut = 1 / 36  # 高频截止频率，以周期为45的高频信号为界
surge_low, surge_high = split_high_low_frequencies(data_list[:, 0], highcut, fs)
sway_low, sway_high = split_high_low_frequencies(data_list[:, 1], highcut, fs)
# Fairlead 的高、低频分量
fairlead_low, fairlead_high = split_high_low_frequencies(data_list[:, -1], highcut, fs)
# plt.plot(surge_high[0:1500])
# plt.show()
# 组合不同的输入
data_low = np.column_stack([surge_low, sway_low])
data_high = np.column_stack([surge_high, sway_high, data_scaled[:, 2], data_scaled[:, 3], data_scaled[:, 4], data_scaled[:, 5]])

df_low = pd.DataFrame(
    {
        'surge_low': surge_low,
        "sway_low": sway_low,
        "fairlead_low": fairlead_low
    }
)
df_high = pd.DataFrame(
    {
        'surge_high':surge_high,
        "sway_high": sway_high,
        "fairlead_high": fairlead_high
    }
)
start = 5000
end = 7000
plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(data_list[start:end, 1], label='Original Signal', color='green')
plt.legend()
plt.title('Original Signal')

plt.subplot(3, 1, 2)
plt.plot(sway_low[start:end], label='Low Signal', color='green')
plt.legend()
plt.title('Low Signal')

plt.subplot(3, 1, 3)
plt.plot(sway_high[start:end], label='High Signal', color='green')
plt.legend()
plt.title('High Signal')
plt.show()

df_low.to_excel('data0545_30_low.xlsx', index=False)
df_high.to_excel('data0545_30_high.xlsx', index=False)
print("succeess")
# print(data_low.shape)
# def create_dataset(data_scaled, lookback):
#     X, y = [], []
#     for i in range(len(data_scaled) - lookback):
#         X.append(data_scaled[i:i + lookback, :])
#         y.append(data_scaled[i + lookback, -1])
#     return np.array(X), np.array(y)


# class TCNLayer(nn.Module):
#     def __init__(self, input_size, output_size, kernel_size=2, dilation=1):
#         super(TCNLayer, self).__init__()
#         self.conv = nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size-1) * dilation, dilation=dilation)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         return self.activation(self.conv(x))
#
# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size=2, num_layers=5):
#         super(TCN, self).__init__()
#         layers = []
#         for i in range(num_layers):
#             dilation = 2 ** i  # Exponential dilation
#             layers.append(TCNLayer(input_size if i == 0 else num_channels, num_channels, kernel_size, dilation))
#         self.network = nn.Sequential(*layers)
#         self.fc = nn.Linear(num_channels, output_size)
#
#     def forward(self, x):
#         # Transpose input to (batch_size, channels, seq_length)
#         # x = x.permute(0, 2, 1)
#         x = self.network(x)
#         # Take the last time step output
#         x = x[:, :, -1]  # (batch_size, num_channels, seq_length)
#         x = self.fc(x)  # (batch_size, output_size)
#         return x
#
# # 预处理高、低频数据集
# X_low, _ = create_dataset(data_low, lookback)
# X_high, _ = create_dataset(data_high, lookback)
# y_low = fairlead_low[lookback:]
# y_high = fairlead_high[lookback:]
#
# # Determine the number of samples
# num_samples_low = X_low.shape[0]
# num_samples_high = X_high.shape[0]
#
# # Reshape inputs to have the correct channel dimension
# X_low = X_low.reshape(num_samples_low, lookback, 2)  # (num_samples, lookback, 2)
# X_high = X_high.reshape(num_samples_high, lookback, 6)  # (num_samples, lookback, 6)
#
# # Convert to PyTorch tensors
# X_low_train = torch.tensor(X_low, dtype=torch.float32).to(device)
# X_high_train = torch.tensor(X_high, dtype=torch.float32).to(device)
#
# # 划分训练和测试集
# X_low_train, X_low_test, y_low_train, y_low_test = train_test_split(X_low, y_low, test_size=0.2, shuffle=False)
# X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(X_high, y_high, test_size=0.2, shuffle=False)
#
# # 转换为 PyTorch tensors
# X_low_train = torch.tensor(X_low_train, dtype=torch.float32).to(device)
# X_low_test = torch.tensor(X_low_test, dtype=torch.float32).to(device)
# y_low_train = torch.tensor(y_low_train.reshape(-1, 1), dtype=torch.float32).to(device)
# y_low_test = torch.tensor(y_low_test.reshape(-1, 1), dtype=torch.float32).to(device)
#
# X_high_train = torch.tensor(X_high_train, dtype=torch.float32).to(device)
# X_high_test = torch.tensor(X_high_test, dtype=torch.float32).to(device)
# y_high_train = torch.tensor(y_high_train.reshape(-1, 1), dtype=torch.float32).to(device)
# y_high_test = torch.tensor(y_high_test.reshape(-1, 1), dtype=torch.float32).to(device)
#
# # 构建两个 TCN 模型
# model_low = TCN(input_size=2, output_size=1, num_channels=64, kernel_size=2, num_layers=5).to(device)
# model_high = TCN(input_size=6, output_size=1, num_channels=64, kernel_size=2, num_layers=5).to(device)
#
# def mean_absolute_percentage_error(y_true, y_pred):
#     epsilon = 1e-8  # 防止除零错误
#     return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
#
# def inverse_transform(scaler, data):
#     return scaler.inverse_transform(data.reshape(-1, 1)).ravel()
#
# # Adjust the train_model function to save model parameters after training
# def train_model(model, X_train, y_train, X_test, y_test, model_name, n_epochs=30, print_interval=5):
#     criterion = nn.MSELoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.002)
#     dataset = TensorDataset(X_train, y_train)
#     loader = DataLoader(dataset, batch_size=10, shuffle=True)
#
#     for epoch in range(n_epochs):
#         model.train()
#         for X_batch, y_batch in loader:
#             optimizer.zero_grad()
#             y_pred = model(X_batch.permute(0, 2, 1))  # Ensure the input is permuted correctly
#             loss = criterion(y_pred, y_batch)
#             loss.backward()
#             optimizer.step()
#
#         # Calculate and output RMSE and MAPE
#         if epoch % print_interval == 0:
#             model.eval()
#             with torch.no_grad():
#                 y_pred_train = model(X_train.permute(0, 2, 1))
#                 train_rmse = torch.sqrt(criterion(y_pred_train, y_train)).item()
#                 train_mape = mean_absolute_percentage_error(y_train, y_pred_train).item()
#
#                 y_pred_test = model(X_test.permute(0, 2, 1))
#                 test_rmse = torch.sqrt(criterion(y_pred_test, y_test)).item()
#                 test_mape = mean_absolute_percentage_error(y_test, y_pred_test).item()
#
#             print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Train MAPE = {train_mape:.2f}% | "
#                   f"Test RMSE = {test_rmse:.4f}, Test MAPE = {test_mape:.2f}%")
#
#     # Save the model parameters
#     torch.save(model.state_dict(), f"{model_name}.pth")
#     print(f"{model_name} parameters saved.")
#
# # Train and save the low-frequency model
# # train_model(model_low, X_low_train, y_low_train, X_low_test, y_low_test, "model_low")
#
# # Train and save the high-frequency model
# train_model(model_high, X_high_train, y_high_train, X_high_test, y_high_test, "model_high")
#
# # 综合预测 with inverse normalization
# with torch.no_grad():
#     y_low_pred = model_low(X_low_test).cpu().numpy().ravel()
#     y_high_pred = model_high(X_high_test).cpu().numpy().ravel()
#
#     # Apply inverse transformation
#     y_low_pred_inv = inverse_transform(scalers[-1], y_low_pred)
#     y_high_pred_inv = inverse_transform(scalers[-1], y_high_pred)
#
#     # Combine predictions
#     y_combined_pred = y_low_pred_inv + y_high_pred_inv
#
#     # Inverse transform the original test values for comparison
#     y_combined_test = inverse_transform(scalers[-1], y_low_test.cpu().numpy().ravel() + y_high_test.cpu().numpy().ravel())
#
# # Calculate RMSE and plot
# rmse_value = np.sqrt(mean_squared_error(y_combined_test, y_combined_pred))
# mape_value = mean_absolute_percentage_error(torch.tensor(y_combined_test), torch.tensor(y_combined_pred))
#
# plt.figure(figsize=(10, 6))
# plt.plot(y_combined_test[::4], label='Original Data', color='blue')
# plt.plot(y_combined_pred[::4], label='Predicted Data (TCN)', color='red')
# plt.text(0.05, 0.95, f'RMSE: {rmse_value:.4f}\nMAPE: {mape_value:.2f}%',
#          transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
#          bbox=dict(facecolor='white', alpha=0.5))
# plt.legend()
# plt.title("Predicted vs Original Data with RMSE and MAPE")
# plt.show()
