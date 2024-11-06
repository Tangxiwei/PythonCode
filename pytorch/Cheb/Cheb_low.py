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
from numpy.polynomial.chebyshev import Chebyshev
# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

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

fs = 20.0  # 采样频率
highcut = 1 / 100  # 高频截止频率，以周期为90的高频信号为界,适当放大
surge_low, surge_high = split_high_low_frequencies(data_list[:, 0], highcut, fs)
sway_low, sway_high = split_high_low_frequencies(data_list[:, 1], highcut, fs)
# Fairlead 的高、低频分量
fairlead_low, fairlead_high = split_high_low_frequencies(data_list[:, -1], highcut, fs)

data_low = np.column_stack([surge_low, sway_low])
data_high = np.column_stack([surge_high, sway_high, data_scaled[:, 2], data_scaled[:, 3], data_scaled[:, 4], data_scaled[:, 5]])

xi = np.sqrt(surge_low**2 + sway_low**2)
y_data = fairlead_low
split_index = int(0.8 * len(y_data))
data_train, data_test = y_data[:split_index], y_data[split_index:]

# pitch_train = data_list[:split_index, 4] * np.pi / 180
# yaw_train = data_list[:split_index, 5] * np.pi / 180
# pitch_test = data_list[split_index:,4]* np.pi/ 180
# yaw_test = data_list[split_index:, 5] * np.pi / 180
# xi_train = data_list[:split_index, 0] + 40.868 * (2 - np.cos(pitch_train) - np.cos(yaw_train))
# xi_test = data_list[split_index:, 0] + 40.868 * (2 - np.cos(pitch_test) - np.cos(yaw_test))
xi_train = xi[:split_index]
xi_test = xi[split_index:]
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
degrees_cheb = [6, 8, 10, 12, 14]
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