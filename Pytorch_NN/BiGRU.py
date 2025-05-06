import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib.font_manager import FontProperties
# from scipy.signal import savgol_filter
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 宋体字体路径
font_chinese = FontProperties(fname=font_path,size=18)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# Read data
file_path = r"D:\datasets\data0545.xlsx"
datasets = pd.read_excel(file_path, header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]

# Data truncation and normalization
truncation = 8000
data_list = datasets.iloc[truncation:].to_numpy()
scalers = [MinMaxScaler() for _ in datasets.columns]
data_scaled = np.column_stack(
    [scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

# Extract features and target signal
features = data_scaled[:, :-1]  # All columns except the last one (fairlead)
target = data_scaled[:, -1]  # Last column (fairlead)

# Create dataset
def create_dataset(features, target, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback, :])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)

lookback = 6
X, y = create_dataset(features, target, lookback)

# Split dataset
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

hidden_size = 96

class BiGRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiGRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define BiGRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Output layer (input size is hidden_size * 2 because of bidirectional GRU)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional

        # Get BiGRU outputs
        out, _ = self.gru(x, h0)

        # Pass the final output through a fully connected layer
        out = self.fc(out[:, -1, :])  # Use the last time step output
        return out

model = BiGRUNetwork(input_size=6, hidden_size=128, output_size=1, num_layers=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=16)
#
# # Learning rate scheduler
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
#
# # Early stopping
patience = 100
best_loss = float('inf')
epochs_no_improve = 0

n_epochs = 40
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)

        # Ensure both y_pred and y_batch have the same shape
        loss = loss_fn(y_pred.squeeze(), y_batch)  # Squeeze y_pred to match y_batch shape
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        val_loss = loss_fn(y_pred, y_test)
        # scheduler.step(val_loss)  # Update learning rate based on validation loss

    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model_weights_0545_BiGRU.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'Early stopping! No improvement for {patience} epochs.')
            break

    # Print training and validation loss
    if epoch % 5 == 0:
        train_loss = loss_fn(model(X_train).squeeze(), y_train).item()
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
torch.save(model.state_dict(), 'best_model_weights_0545_BiGRU.pth')
# Load the best model
# model.load_state_dict(torch.load('best_model_weights_0545_BiGRU.pth'))

# Predict and inverse transform
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
    y_test_original = scalers[-1].inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).ravel()
    y_pred_inv = y_pred_inv / 1000
    y_test_original = y_test_original / 1000

    # Calculate evaluation metrics
    test_rmse_value = np.sqrt(np.mean((y_pred_inv - y_test_original) ** 2))
    mae_value = np.mean(np.abs(y_pred_inv - y_test_original))
    r2_value = r2_score(y_test_original, y_pred_inv)
    test_mape_value = np.mean(np.abs((y_pred_inv - y_test_original) / y_test_original)) * 100
    print(f"Test RMSE: {test_rmse_value:.4f}, R2: {r2_value:.4f}, MAE: {mae_value:.4f}, MAPE: {test_mape_value:.2f}%")

# file_path = "output952.xlsx"  # 这一部分是将数据输出
# df = pd.read_excel(file_path)
# df.insert(10, 'BiGRU', y_pred_inv)  # 在第二列（索引为1的位置）插入数据
# # # 将修改后的 DataFrame 写回 Excel 文件（覆盖原文件）
# df.to_excel(file_path, index=False)  # index=False 表示不保存索引列

# Plot results
x = np.arange(0, len(y_pred_inv) * 0.05, 0.05)  # 0.2 is the time interval
x = x[1:]
plt.figure(figsize=(12, 10.5))
plt.ylim(500, 4500)
ticks = np.arange(500, 4600, 500)  # 注意：arange 的结束值是开区间，所以要加 1
plt.yticks(ticks)
plt.tick_params(axis='both', which='major', labelsize=24)
# plt.plot(x_new,y_test_original[lookback:], label='Original Data', color='blue', linewidth=1)
# plt.plot(x_new,y_pred_inv, label='Predicted Data (CNN+LSTM+Attention)', color='red', linestyle='--', linewidth=1.8)
plt.plot(x,y_pred_inv, label='预测值 (BiGRU)',color='#FF5733', linestyle='--', linewidth=2)
plt.plot(x,y_test_original, label='真实值', color='#2E86C1', linewidth=2)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('时间 (秒)', fontsize=24, fontproperties = font_chinese)
plt.ylabel('张力 (千牛)', fontsize=24, fontproperties = font_chinese)
plt.legend(  prop={
        'size': 24,          # 字体大小
        'family': 'SimSun',  # 字体族
    })
plt.text(0.55, 0.92, f'MAPE: {test_mape_value:.2f}%', transform=plt.gca().transAxes, fontsize=26,
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
# plt.savefig("cn_bigru952.png", dpi = 200 )
plt.show()