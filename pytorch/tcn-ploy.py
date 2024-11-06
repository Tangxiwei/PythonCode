import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Load dataset
datasets = pd.read_excel("D:/datasets/data.1/data0545_30.xlsx", header=0)
datasets.columns = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
truncation = 8000
lookback = 8
dims = ["surge", "sway", "heave", "roll", "pitch", "yaw", "fairlead"]
data_list = datasets[dims].iloc[truncation:].to_numpy()

# Normalize data
scalers = [MinMaxScaler() for _ in dims]
data_scaled = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

def create_dataset(data_scaled, lookback):
    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i + lookback, :-1])
        y.append(data_scaled[i + lookback, -1])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, lookback)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

class TCNLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=2, dilation=1):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size-1) * dilation, dilation=dilation)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, num_layers=5):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            layers.append(TCNLayer(input_size if i == 0 else num_channels, num_channels, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, output_size)

    def forward(self, x):
        # Transpose input to (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        # Take the last time step output
        x = x[:, :, -1]  # (batch_size, num_channels, seq_length)
        x = self.fc(x)  # (batch_size, output_size)
        return x

# Instantiate model
model = TCN(input_size=6, output_size=1, num_channels=64, kernel_size=2, num_layers=5).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
train_dataset = TensorDataset(X_train, y_train)
loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Training
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
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
        print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

# Predictions
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()
    y_test_original = data_list[:, -1][-len(y_test):]
rmse_value = np.sqrt(mean_squared_error(y_test_original, y_pred_inv))
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test_original[::4], label='Original Data', color='blue')
plt.plot(y_pred_inv[::4], label='Predicted Data (TCN)', color='red')
plt.text(0.05, 0.95, f'RMSE: {rmse_value:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.show()
