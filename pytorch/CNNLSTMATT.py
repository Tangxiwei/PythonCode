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
datasets = pd.read_excel(r"D:\datasets\data..5\13.1_14.9.xlsx", header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
# 数据截断和缩放
truncation = 0
lookback = 6
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
split_index = int(0.8 * len(y_data))
data_train, data_test = y_data[:split_index], y_data[split_index:]

# 定义注意力机制
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
class ConvLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.5):
        super(ConvLSTMAttention, self).__init__()
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
# 实例化模型
model = ConvLSTMAttention(input_size=6, hidden_size=256, output_size=1, num_layers=3, dropout_prob=0.5).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
train_dataset = TensorDataset(X_train, y_train)
loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
n_epochs = 40
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
            y_pred_test = model(X_test)
            test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
        print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")
torch.save(model.state_dict(), 'model_weights_13.1_lstm.pth')
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
    y_test_original = data_list[:, -1][-len(y_test):]
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Data', color='blue')
plt.plot(y_pred_inv, label='Predicted Data(LSTM)', color='red')
plt.legend()
plt.show()

