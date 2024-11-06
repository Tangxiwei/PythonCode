import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from numpy.polynomial.chebyshev import Chebyshev
# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# datasets = pd.read_excel("D:/datasets/fenli/data0545_30_high.xlsx", header=0)
# datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
truncation = 0
# lookback = 8
dims = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
# data_list = datasets[dims].iloc[truncation:].to_numpy()
# # Normalize data
# scalers = [MinMaxScaler() for _ in dims]
# data_scaled = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

datasets_origin = pd.read_excel("D:/datasets/fenli/data0545_30_high.xlsx", header=0)
datasets_origin.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
data_list_origin = datasets_origin[dims].iloc[truncation:].to_numpy()
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

fs = 10.0  # 采样频率
highcut = 1 / 55  # 高频截止频率，以周期为90的高频信号为界,适当放大

fairlead_low, fairlead_high = split_high_low_frequencies(data_list_origin[:, -1], highcut, fs)

# data_high_scaled = data_scaled[:,:-1]
# fairlead_high_scaled = data_scaled[:,-1]

y_data = fairlead_low
split_index = int(0.8 * len(y_data))
data_train, data_test = y_data[:split_index], y_data[split_index:]
pitch_train = data_list_origin[:split_index, 4] * np.pi / 180
yaw_train = data_list_origin[:split_index, 5] * np.pi / 180
pitch_test = data_list_origin[split_index:,4]* np.pi/ 180
yaw_test = data_list_origin[split_index:, 5] * np.pi / 180
xi_train = data_list_origin[:split_index, 0] + 40.868 * (2 - np.cos(pitch_train) - np.cos(yaw_train))
xi_test = data_list_origin[split_index:, 0] + 40.868 * (2 - np.cos(pitch_test) - np.cos(yaw_test))

# Scaling function to transform data to [-1, 1]
def scale_to_unit_interval(data, min_val, max_val):
    return 2 * (data - min_val) / (max_val - min_val) - 1
def inverse_scale_from_unit_interval(scaled_data, min_val, max_val):
    # Inverse transformation to go from [-1, 1] back to original range
    return 0.5 * (scaled_data + 1) * (max_val - min_val) + min_val
# Scale xi_train and xi_test to [-1, 1]
xi_min, xi_max = xi_train.min(), xi_train.max()
xi_train_scaled = scale_to_unit_interval(xi_train, xi_min, xi_max)
xi_test_scaled = scale_to_unit_interval(xi_test, xi_min, xi_max)
# Scale y_data to [-1, 1]
y_min, y_max = y_data.min(), y_data.max()
y_data_scaled = scale_to_unit_interval(data_train, y_min, y_max)
degrees_cheb = [6, 8, 10, 12, 14, 18]
fits_cheb= []
mse_list_cheb = []
r2_list_cheb = []
for degree in degrees_cheb:
    # Fit Chebyshev polynomial to the scaled y_data
    coefs = Chebyshev.fit(xi_train_scaled, y_data_scaled, degree)
    fits_cheb.append(coefs)
    # Predict on the scaled training data
    y_fit_scaled = coefs(xi_train_scaled)
    # Calculate MSE and R2 using scaled y
    mse = mean_squared_error(y_data_scaled, y_fit_scaled)
    r2 = r2_score(y_data_scaled, y_fit_scaled)
    mse_list_cheb.append(mse)
    r2_list_cheb.append(r2)
    print(f"Degree {degree}: MSE = {mse}, R² = {r2}")
# Choose the best fit
best_degree = degrees_cheb[np.argmin(mse_list_cheb)]
best_fit = fits_cheb[np.argmin(mse_list_cheb)]
print(f"Best degree: {best_degree}")
print(f"Coefficients: {best_fit.convert().coef}")
# Predict on the scaled test data
y_test_fit_scaled = best_fit(xi_test_scaled)
y_test_fit_original = inverse_scale_from_unit_interval(y_test_fit_scaled, y_min, y_max)
plt.figure(figsize=(10, 6))
plt.plot(y_test_fit_original,label = 'Cheb Poly Fit Low',color = 'green')
plt.plot(data_test, label = 'Fairlead Low',color = 'darkblue')
plt.show()

# # 定义注意力机制
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn_weights = nn.Linear(hidden_size, hidden_size)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, lstm_output):
#         attn_scores = self.attn_weights(lstm_output)
#         attn_scores = self.softmax(attn_scores)
#         context = torch.sum(attn_scores * lstm_output, dim=1)
#         return context, attn_scores
#
# # 定义1D卷积+LSTM+注意力模型
# class ConvLSTMWithAttention(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.5):
#         super(ConvLSTMWithAttention, self).__init__()
#         self.hidden_size = hidden_size
#
#         # 1D卷积层
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#
#         # 线性层将卷积层输出映射到LSTM输入尺寸
#         self.fc = nn.Linear(128, hidden_size)
#
#         # LSTM层
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#
#         # 注意力机制
#         self.attention = Attention(hidden_size)
#
#         # 全连接层
#         self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # 1D卷积层
#         x = x.permute(0, 2, 1)  # Change the shape to (batch, features, sequence_length)
#         conv_out = self.conv(x)
#
#         # 线性层映射
#         conv_out = conv_out.permute(0, 2, 1)  # Change the shape back to (batch, sequence_length, features)
#         mapped = self.fc(conv_out)
#
#         # LSTM
#         lstm_out, _ = self.lstm(mapped)
#
#         # 注意力机制
#         context, attn_scores = self.attention(lstm_out)
#
#         # 全连接层
#         x = self.linear(context)
#         return x, attn_scores
#
# def create_dataset(data_scaled, lookback):
#     X, y = [], []
#     for i in range(len(data_scaled) - lookback):
#         X.append(data_scaled[i:i + lookback, :])
#         y.append(data_scaled[i + lookback, -1])
#     return np.array(X), np.array(y)
#
# # 预处理高、低频数据集
# X_high, _ = create_dataset(data_high_scaled, lookback)
# y_high = fairlead_high_scaled[lookback:]
#
# # Determine the number of samples
# num_samples_high = X_high.shape[0]
#
# # Reshape inputs to have the correct channel dimension
# X_high = X_high.reshape(num_samples_high, lookback, 6)  # (num_samples, lookback, 6)
#
# # Convert to PyTorch tensors
# X_high_train = torch.tensor(X_high, dtype=torch.float32).to(device)
#
# # 划分训练和测试集
# X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(X_high, y_high, test_size=0.2, shuffle=False)
#
# # 转换为 PyTorch tensors
# X_high_train = torch.tensor(X_high_train, dtype=torch.float32).to(device)
# X_high_test = torch.tensor(X_high_test, dtype=torch.float32).to(device)
# y_high_train = torch.tensor(y_high_train.reshape(-1, 1), dtype=torch.float32).to(device)
# y_high_test = torch.tensor(y_high_test.reshape(-1, 1), dtype=torch.float32).to(device)
#
# model_high  = ConvLSTMWithAttention(input_size=6, hidden_size=256, output_size=1, num_layers=3, dropout_prob=0.5).to(device)
# model_high.load_state_dict(torch.load('model_weights_ConvLSTM_Attention_high0545_30.pth'))
# model_high.eval()
# with torch.no_grad():
#     y_pred, _ = model_high(X_high_test)
#     y_pred = y_pred.squeeze()
#     y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
#
# cut_len = 200
# y_pred_inv = y_pred_inv[:-cut_len]
# y_pred_sum = y_pred_inv + y_test_fit_original[:-cut_len]
# y_test_original = data_list[:, -1][split_index:-cut_len]
#
# test_mape_value = np.mean(np.abs((y_pred_sum - y_test_original[lookback:]) / y_test_original[lookback:])) * 100
#
# # Plot results with enhanced aesthetics
# plt.figure(figsize=(12, 7))
# plt.plot(y_pred_inv, label='Predicted Data (Conv+LSTM+Attention+Filter+Cheb)', color='#FF5733', linestyle='--', linewidth=2)
# plt.plot(y_test_original[lookback:], label='Original Data', color='#2E86C1', linewidth=2)
# plt.xlabel('Time (0.05s)', fontsize=14)
# plt.ylabel('Fairlead Tension (N)', fontsize=14)
# # plt.title('Fairlead Tension Prediction with ConvLSTM+Attention Model', fontsize=16, weight='bold')
# plt.legend(fontsize=12)
# # plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
# # Annotate RMSE and MAPE on the plot
# # plt.text(0.1, 0.9, f'Test RMSE: {test_rmse_value:.4f}', transform=plt.gca().transAxes, fontsize=12,
# #          bbox=dict(facecolor='white', alpha=0.8))
# plt.text(0.02, 0.95, f'Test MAPE: {test_mape_value:.2f}%', transform=plt.gca().transAxes, fontsize=12,
#          bbox=dict(facecolor='white', alpha=0.8))
#
# plt.tight_layout()
# # plt.savefig("ConvLSTMATT_13.1hs1.78.png", dpi=400)  # Optional save
# plt.show()
