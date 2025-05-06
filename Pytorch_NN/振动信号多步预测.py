import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split

# 字体设置
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
font_chinese = FontProperties(fname=font_path, size=18)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 数据加载与预处理
def load_and_preprocess_data(file_path, lookback=15, pre_step=3):
    # 读取数据
    datasets = pd.read_excel(file_path, header=0)
    datasets.columns = ["Dynamic x"]

    # 更新特征列
    dims = ["Dynamic x"]
    data_list = datasets[dims].to_numpy()

    # 归一化
    scalers = [MinMaxScaler() for _ in dims]
    data_scaled = np.column_stack([
        scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel()
        for i, scaler in enumerate(scalers)
    ])

    # 创建时间序列数据集
    X, y = [], []
    for i in range(len(data_scaled) - lookback - pre_step + 1):
        X.append(data_scaled[i:i + lookback, 0])
        y.append(data_scaled[i + lookback:i + lookback + pre_step, 0])

    # 转换为张量并固定形状
    X = np.array(X)  # shape: (n_samples, lookback)
    y = np.array(y)  # shape: (n_samples, pre_step)

    return X, y, data_list[:, 0], scalers[0]


# 加载数据
file_path = r"D:\usertemp\dataT10.xlsx"
lookback = 15
pre_step = 3
X, y, y_raw, y_scaler = load_and_preprocess_data(file_path, lookback, pre_step)

# 数据集划分 (保持形状)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# 转换为PyTorch张量 (形状已固定)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # (n_train, lookback)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # (n_test, lookback)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)  # (n_train, pre_step)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)  # (n_test, pre_step)

# 原始数据划分 (用于最终评估)
split_index = int(0.8 * len(X))
y_test_raw = y_raw[split_index + lookback + pre_step - 1:]


# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # shape: (batch, lookback, hidden_size)
        last_output = lstm_out[:, -1, :]  # shape: (batch, hidden_size)
        return self.linear(last_output)  # shape: (batch, output_size)


# 训练配置
model = LSTMModel(input_size=1, hidden_size=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 训练循环
best_rmse = float('inf')
patience = 10
no_improve = 0

for epoch in range(70):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch[:, 0].unsqueeze(1))  # 只预测第一步
        loss.backward()
        optimizer.step()

    # 验证
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)

            train_rmse = torch.sqrt(criterion(y_pred_train, y_train[:, 0].unsqueeze(1))).item()
            test_rmse = torch.sqrt(criterion(y_pred_test, y_test[:, 0].unsqueeze(1))).item()

        print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

        # Early Stopping
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            no_improve = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered!")
                break


# 加载最佳模型
model.load_state_dict(torch.load('best_lstm_model.pth'))

# 多步预测函数
def iterative_forecast(model, initial_seq, steps=pre_step):
    model.eval()
    predictions = []
    current_seq = initial_seq.clone()

    for _ in range(steps):
        with torch.no_grad():
            pred = model(current_seq.unsqueeze(0))[0, 0].item()
        predictions.append(pred)
        # 更新输入序列：滑动窗口+用预测值作为新输入
        current_seq = torch.cat([current_seq[1:], torch.tensor([pred], device=device).unsqueeze(0)])

    return predictions


# 预测整个测试集
all_predictions = []
for i in range(len(X_test)):
    initial_seq = X_test[i]
    future_preds_normalized = iterative_forecast(model, initial_seq, steps=pre_step)
    all_predictions.append(future_preds_normalized)

# 将预测结果转换为原始尺度
all_predictions = np.array(all_predictions)
all_predictions_inv = y_scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(all_predictions.shape)

# 计算指标
r2 = r2_score(y_test_raw[:len(all_predictions_inv)], all_predictions_inv[:, -1])
mape = np.mean(np.abs((all_predictions_inv[:, -1] - y_test_raw[:len(all_predictions_inv)]) / y_test_raw[:len(all_predictions_inv)])) * 100

# 可视化
plt.figure(figsize=(12, 7))
x_axis = np.arange(len(all_predictions_inv)) * 0.2  # 假设时间间隔为0.2秒
plt.plot(x_axis, all_predictions_inv[:, -1], '--', label='Predicted', linewidth=2)
plt.plot(x_axis, y_test_raw[:len(all_predictions_inv)], '-', label='True', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Dynamic x (mm)', fontsize=14)
plt.legend(prop={'size': 12})
plt.grid(linestyle='--', alpha=0.7)
plt.text(0.6, 0.9, f'R²: {r2:.4f}\nMAPE: {mape:.2f}%', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig("lstm_prediction.png", dpi=300)
plt.show()