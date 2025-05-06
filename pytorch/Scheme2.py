import torch
import torch.nn as nn
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
# 读取数据
datasets = pd.read_excel(r"D:\datasets\data91116.xlsx", header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
# 数据截断和缩放
truncation = 0
lookback = 8
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
y_data = data_list[:,6]
split_index = len(X_train)
data_train, data_test = y_data[:split_index], y_data[split_index:]
# print(split_index,0.8 * len(y_data))
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
# Fit Chebyshev polynomial as before but on scaled data
from numpy.polynomial.chebyshev import Chebyshev
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
best_degree = degrees_cheb[np.argmin(mse_list)]
best_fit = fits_cheb[np.argmin(mse_list)]
print(f"Best degree: {best_degree}")
print(f"Coefficients: {best_fit.convert().coef}")
# Predict on the scaled test data
y_test_fit_scaled = best_fit(xi_test_scaled)
y_test_fit_original = inverse_scale_from_unit_interval(y_test_fit_scaled, y_min, y_max)
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attn_scores = self.attn_weights(lstm_output)
        attn_scores = self.softmax(attn_scores)
        context = torch.sum(attn_scores * lstm_output, dim=1)
        return context, attn_scores
# 定义1D卷积+LSTM+注意力模型
class ConvLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.5):
        super(ConvLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size

        # 1D卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 线性层将卷积层输出映射到LSTM输入尺寸
        self.fc = nn.Linear(128, hidden_size)

        # LSTM层
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # 注意力机制
        self.attention = Attention(hidden_size)

        # 全连接层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 1D卷积层
        x = x.permute(0, 2, 1)  # Change the shape to (batch, features, sequence_length)
        conv_out = self.conv(x)

        # 线性层映射
        conv_out = conv_out.permute(0, 2, 1)  # Change the shape back to (batch, sequence_length, features)
        mapped = self.fc(conv_out)

        # LSTM
        lstm_out, _ = self.lstm(mapped)

        # 注意力机制
        context, attn_scores = self.attention(lstm_out)

        # 全连接层
        x = self.linear(context)
        return x, attn_scores
model = ConvLSTMWithAttention(input_size=6, hidden_size=256, output_size=1, num_layers=3, dropout_prob=0.1).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
train_dataset = TensorDataset(X_train, y_train)
loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
n_epochs = 40
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
            y_pred_test = model(X_test)
            test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
        print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")
torch.save(model.state_dict(), 'model_weights_91116_lstm.pth')
# model = ConvLSTMWithAttention(input_size=6, hidden_size=256, output_size=1, num_layers=3, dropout_prob=0.5).to(device)
# model.load_state_dict(torch.load('model_weights_ConvLSTM_Attention_178.pth'))
model.eval()
# 预测并反归一化
with torch.no_grad():
    y_pred, _ = model(X_test)
    y_pred = y_pred.squeeze()
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
    y_test_original = data_list[:, -1][-len(y_test):]
# lower_bound = 0.99*y_test_fit_original
# upper_bound = 1.01*y_test_fit_original
# y_pred_inv_poly = np.clip(y_pred_inv,lower_bound[lookback:],upper_bound[lookback:])
# test_mape_value = np.mean(np.abs((y_pred_inv_poly - y_test_original) / y_test_original)) * 100
# 绘制预测结果
fairlead = y_test_original
fairlead = fairlead/1000
fairlead_fit = y_pred_inv/1000
x = np.arange(0, len(fairlead) * 0.05, 0.05)
plt.figure(figsize=(12, 7))
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Fairlead Tension (kN)', fontsize=14)
# # plt.title('Fairlead Tension Prediction with ConvLSTM+Attention Model', fontsize=16, weight='bold')
# plt.legend(fontsize=12)
# plt.plot(y_test_fit_original, label = 'Cheb Poly Fit',color = 'red')
# plt.plot(y_test_fit, label = 'Poly Fit',color = 'black')
plt.plot(x, fairlead_fit, label = "ANN Fitting", color = 'red')
plt.plot(x,fairlead, label='Original Tension Data', color='blue')
# plt.text(0.02, 0.95, f'Test MAPE: {test_mape_value:.2f}%', transform=plt.gca().transAxes, fontsize=12,
#          bbox=dict(facecolor='white', alpha=0.8))
# plt.plot(y_pred_inv, label='Predicted Data(LSTM)', color='red')
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, fancybox=True, framealpha=0.9)
plt.show()

