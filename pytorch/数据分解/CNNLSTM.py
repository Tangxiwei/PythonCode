import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PyEMD import EMD

# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# 读取数据
file_path = "D:/datasets/data0545.xlsx"
datasets = pd.read_excel(file_path, header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]

# 数据截断和归一化
truncation = 8000
data_list = datasets.iloc[truncation:].to_numpy()
scalers = [MinMaxScaler() for _ in datasets.columns]
data_scaled = np.column_stack(
    [scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

# 提取特征和目标信号
features = data_scaled[:, :-1]  # 所有列除了最后一列fairlead
target = data_scaled[:, -1]  # 最后一列fairlead

# 进行EMD分解
emd = EMD()
IMFs = emd(target)


# 创建数据集
def create_dataset(features, IMFs, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback, :])
        y.append(IMFs[:, i + lookback])  # 预测IMFs
    return np.array(X), np.array(y)


lookback = 8
X, y = create_dataset(features, IMFs, lookback)

# 数据集划分
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


# 定义CNN+LSTM模型
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.5):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_prob)

        # LSTM layer
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)  # Change the shape to (batch, features, sequence_length)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # LSTM
        x = x.permute(0, 2, 1)  # Change the shape back to (batch, sequence_length, features)
        x, _ = self.lstm(x)

        # Fully connected layer
        x = self.dropout(x[:, -1, :])
        x = self.linear(x)
        return x


# 实例化模型
input_size = X_train.shape[2]  # 特征数量
output_size = IMFs.shape[0]  # IMF数量
model = CNNLSTMModel(input_size=input_size, hidden_size=256, output_size=output_size, num_layers=3,
                     dropout_prob=0.5).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
train_dataset = TensorDataset(X_train, y_train)
loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 模型训练
n_epochs = 45
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
torch.save(model.state_dict(), 'model_weights_0545_EMD_CNNLSTM.pth')

# 预测并反归一化
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_sum = np.sum(y_pred.cpu().numpy(), axis=1)
    y_pred_inv = scalers[-1].inverse_transform(y_pred_sum.reshape(-1, 1)).ravel()
    y_test_original = data_list[:, -1][split_index:]
print(y_pred_inv.shape)

# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_pred_inv, label='Predicted Data (CNN+LSTM+EMD)', color='red')
plt.plot(y_test_original, label='Original Data', color='blue')
plt.legend()
plt.show()
