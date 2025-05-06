import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
lookback = 8
# 读取多个数据文件
file_paths = ["D:/datasets/data0545.xlsx"]

# 数据预处理和创建数据集
def preprocess_data(file_path, truncation=8000, lookback=8):
    datasets = pd.read_excel(file_path, header=0)
    datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
    data_list = datasets.iloc[truncation:].to_numpy()
    scalers = [MinMaxScaler() for _ in datasets.columns]
    data_scaled = np.column_stack(
        [scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])
    features = data_scaled[:, :-1]
    target = data_scaled[:, -1]
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback, :])
        y.append(target[i + lookback])
    X, y = np.array(X), np.array(y)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test, scalers[-1], data_list[:, -1][split_index:]

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
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
        x = x.permute(0, 2, 1)  # (batch, features, sequence_length)
        conv_out = self.conv(x)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, sequence_length, features)
        mapped = self.fc(conv_out)
        lstm_out, _ = self.lstm(mapped)
        context, attn_scores = self.attention(lstm_out)
        x = self.linear(context)
        return x, attn_scores

# 模型训练和预测
def train_and_predict(X_train, X_test, y_train, y_test, scaler, y_test_original, file_index, n_epochs=40):
    input_size = X_train.shape[2]
    output_size = 1
    model = ConvLSTMWithAttention(input_size=input_size, hidden_size=256, output_size=output_size, num_layers=3, dropout_prob=0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    train_dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred, _ = model(X_batch)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_train, _ = model(X_train)
                y_pred_test, _ = model(X_test)
                train_rmse = torch.sqrt(criterion(y_pred_train.squeeze(), y_train))
                test_rmse = torch.sqrt(criterion(y_pred_test.squeeze(), y_test))
            print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

    torch.save(model.state_dict(), f'model_weights_ConvLSTM_Attention_{file_index}.pth')

    model.eval()
    with torch.no_grad():
        y_pred, _ = model(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred_inv, label='Predicted Data (Conv+LSTM+Attention)', color='red')
    plt.plot(y_test_original[lookback:], label='Original Data', color='blue')
    test_rmse_value = np.sqrt(np.mean((y_pred_inv - y_test_original[lookback:]) ** 2))
    plt.text(0.1, 0.9, f'Test RMSE: {test_rmse_value:.4f}', transform=plt.gca().transAxes)
    plt.legend()
    plt.title(f'Results for Dataset {file_index}')
    plt.savefig(f'results_{file_index}.png')
    plt.close()

for i, file_path in enumerate(file_paths):
    X_train, X_test, y_train, y_test, scaler, y_test_original = preprocess_data(file_path)
    X_train, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32).to(device), [X_train, X_test])
    y_train, y_test = map(lambda y: torch.tensor(y, dtype=torch.float32).to(device), [y_train, y_test])
    train_and_predict(X_train, X_test, y_train, y_test, scaler, y_test_original, i + 545)
