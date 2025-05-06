import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# 数据读取和预处理
positions = pd.read_excel(r".\w25\w25positions.xlsx")  # x, y, z
velocities = pd.read_excel(r".\w25\w25velocities.xlsx")  # vx, vy, vz
tensions = pd.read_excel(r".\w25\w25tensions.xlsx")  # Tension
time_series = np.arange(0, len(velocities) * 0.05, 0.05)

# 数据归一化
scaler_positions = MinMaxScaler()
scaler_velocities = MinMaxScaler()
scaler_tensions = MinMaxScaler()
scaler_time = MinMaxScaler()

positions = scaler_positions.fit_transform(positions)
velocities = scaler_velocities.fit_transform(velocities)
tensions = scaler_tensions.fit_transform(tensions)
time_series = scaler_time.fit_transform(time_series.reshape(-1, 1))

# 构建时间序列数据集
def create_dataset(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])  # 历史数据
        y.append(data[i + lookback])  # 目标值
    return np.array(X), np.array(y)

lookback = 10  # 滑动窗口大小
X_pos, _ = create_dataset(positions, lookback)
X_vel, _ = create_dataset(velocities, lookback)
_, y_tension = create_dataset(tensions, lookback)
_, X_time = create_dataset(time_series,lookback)
# 转换为 PyTorch 张量
X_pos = torch.tensor(X_pos, dtype=torch.float32)
X_vel = torch.tensor(X_vel, dtype=torch.float32)
X_time = torch.tensor(X_time, dtype=torch.float32)
y_tension = torch.tensor(y_tension, dtype=torch.float32)

# 划分训练集和测试集
train_size = int(0.8 * len(X_pos))
X_pos_train, X_pos_test = X_pos[:train_size], X_pos[train_size:]
X_vel_train, X_vel_test = X_vel[:train_size], X_vel[train_size:]
X_time_train, X_time_test = X_time[:train_size], X_time[train_size:]
y_tension_train, y_tension_test = y_tension[:train_size], y_tension[train_size:]

# 打印输入形状
print(f"Position Branch input shape: {X_pos_train.shape}")
print(f"Velocity Branch input shape: {X_vel_train.shape}")
print(f"Time Trunk input shape: {X_time_train.shape}")

# 局部注意力分支网络
class ConvAttentionBranch(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, local_window):
        super(ConvAttentionBranch, self).__init__()
        self.conv = nn.Conv1d(input_dim, embed_size, kernel_size=local_window, padding=local_window//2)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        x = self.conv(x)  # 局部卷积
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, embed_size)
        attn_output, _ = self.attention(x, x, x)  # 多头注意力
        return self.fc(attn_output[:, -1, :])  # 取最后一个时间步的输出

# 多头注意力融合
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, branch_pos_output, branch_vel_output):
        combined = torch.cat([branch_pos_output.unsqueeze(1), branch_vel_output.unsqueeze(1)], dim=1)  # 拼接
        attn_output, _ = self.attention(combined, combined, combined)  # 多头注意力
        return self.fc(attn_output.mean(dim=1))  # 取均值

# 改进版 DeepONet
class ImprovedDeepONet(nn.Module):
    def __init__(self, pos_input_dim, vel_input_dim, trunk_input_dim, embed_size, num_heads, local_window):
        super(ImprovedDeepONet, self).__init__()
        self.branch_pos_net = ConvAttentionBranch(pos_input_dim, embed_size, num_heads, local_window)
        self.branch_vel_net = ConvAttentionBranch(vel_input_dim, embed_size, num_heads, local_window)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, embed_size),
            nn.ReLU()
        )
        self.attention_fusion = MultiHeadAttentionFusion(embed_size, num_heads)
        self.output_layer = nn.Linear(embed_size, 1)

    def forward(self, branch_pos_input, branch_vel_input, trunk_input):
        branch_pos_output = self.branch_pos_net(branch_pos_input)
        branch_vel_output = self.branch_vel_net(branch_vel_input)
        trunk_output = self.trunk_net(trunk_input)

        combined_branch = self.attention_fusion(branch_pos_output, branch_vel_output)
        combined = combined_branch * trunk_output
        return self.output_layer(combined)

# 初始化模型
embed_size = 64
num_heads = 4
local_window = 5
model = ImprovedDeepONet(
    pos_input_dim=3, vel_input_dim=3, trunk_input_dim=1,
    embed_size=embed_size, num_heads=num_heads, local_window=local_window
)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# 训练
epochs = 500
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output_pred = model(X_pos_train, X_vel_train, X_time_train)
    train_loss = loss_fn(output_pred, y_tension_train)
    train_loss.backward()
    optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        test_pred = model(X_pos_test, X_vel_test, X_time_test)
        test_loss = loss_fn(test_pred, y_tension_test)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Test Loss = {test_loss.item():.6f}")
torch.save(model.state_dict(), 'model_weights_w25_DeepONetTra.pth')
# 可视化损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Testing Loss")
plt.grid()
plt.show()

# 模型预测
model.eval()
with torch.no_grad():
    predicted_tension = model(X_pos_test, X_vel_test, X_time_test).numpy()

# 反归一化张力
predicted_tension = scaler_tensions.inverse_transform(predicted_tension)
true_tension = scaler_tensions.inverse_transform(y_tension_test.numpy())

# 绘图比较预测值和真实值
plt.figure(figsize=(10, 6))
plt.plot(predicted_tension, label="Predicted Tension", color="red", linestyle="--")
plt.plot(true_tension, label="True Tension", color="blue", linestyle="-")
plt.xlabel("Time Step")
plt.ylabel("Tension")
plt.title("Predicted vs True Tension")
plt.legend()
plt.grid()
plt.show()
