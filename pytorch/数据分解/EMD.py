import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PyEMD import EMD
from sklearn.metrics import mean_squared_error

# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# 读取数据文件路径
file_path = "D:/datasets/data0545.xlsx"

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.linear(x)
        return x

# 数据预处理
def preprocess_data(file_path, truncation=8000, lookback=8):
    datasets = pd.read_excel(file_path, header=0)
    datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
    data_list = datasets.iloc[truncation:].to_numpy()

    # 进行EMD分解
    target = data_list[:, -1]
    emd = EMD()
    IMFs = emd(target)
    print(f"IMFs shape: {IMFs.shape}")

    # 截断EMD分解后的IMFs，使其与特征匹配
    IMFs = IMFs[:, :len(data_list)]
    print(f"Truncated IMFs shape: {IMFs.shape}")

    # 归一化数据
    scalers = [MinMaxScaler() for _ in datasets.columns[:-1]]
    features = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])
    print(f"Features shape: {features.shape}")

    # 归一化IMF分量
    imf_scalers = [MinMaxScaler() for _ in range(IMFs.shape[0])]
    IMFs_scaled = np.column_stack([scaler.fit_transform(IMFs[i, :].reshape(-1, 1)).ravel() for i, scaler in enumerate(imf_scalers)])
    print(f"IMFs_scaled shape: {IMFs_scaled.shape}")

    def create_dataset(features, IMFs_scaled, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            if i + lookback < IMFs_scaled.shape[0]:  # 确保索引不超出范围
                X.append(features[i:i + lookback, :])
                y.append(IMFs_scaled[i + lookback, :])
        return np.array(X), np.array(y)

    X, y = create_dataset(features, IMFs_scaled, lookback)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, imf_scalers, data_list[:, -1][split_index:], IMFs.shape[0]

#模型训练和预测
def train_and_predict(model, X_train, X_test, y_train, y_test, imf_scalers, y_test_original, output_size, n_epochs=45):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    train_dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_train = model(X_train)
                train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
                y_pred_test = model(X_test)
                test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
            print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")

    torch.save(model.state_dict(), 'model_EMD_weights.pth')
    X_train, X_test, y_train, y_test, imf_scalers, y_test_original, output_size = preprocess_data(file_path)
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=256, output_size=output_size, num_layers=3,
                      dropout_prob=0.5).to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_sum = np.sum(y_pred.cpu().numpy(), axis=1)
        y_pred_inv = imf_scalers[0].inverse_transform(y_pred_sum.reshape(-1, 1)).ravel()

    # 绘制预测结果并标注RMSE
    test_error_value = np.abs((y_test_original[8:]-y_pred_inv)/y_test_original[8:])
    error_per = test_error_value.mean()

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred_inv, label='Predicted Data (LSTM+EMD)', color='red')
    plt.plot(y_test_original, label='Original Data', color='blue')
    plt.text(0.1, 0.9, f'Test RMSE: {error_per:.4f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.title('Results')
    plt.show()

# 处理数据文件

train_and_predict(model, X_train, X_test, y_train, y_test, imf_scalers, y_test_original, output_size)
