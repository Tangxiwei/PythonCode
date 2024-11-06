import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set device to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
datasets = pd.read_excel("D:/datasets/data.1/data0545_30_high.xlsx", header=0)
print(f"Loaded dataset with shape: {datasets.shape}")  # Print shape to see if it matches expectations

# Ensure the number of columns matches the expected column names
# Modify the column names according to the actual structure
if datasets.shape[1] == 7:
    datasets.columns = ["surgehigh", "swayhigh", "heave", "roll", "pitch", "yaw", "fairleadhigh"]
else:
    print(f"Warning: Expected 7 columns, but found {datasets.shape[1]}. Adjust the column names accordingly.")
    # Example fallback in case of mismatch
    datasets.columns = [f"feature_{i}" for i in range(datasets.shape[1])]

# Define parameters
truncation = 0
lookback = 8
dims = ["surgehigh", "swayhigh", "heave", "roll", "pitch", "yaw", "fairleadhigh"]
data_list = datasets[dims].iloc[truncation:].to_numpy()

# Normalize data
scalers = [MinMaxScaler() for _ in dims]
data_scaled = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

# Extract features and target
features = data_scaled[:, :-1]  # All columns except the last column (fairlead)
target = data_scaled[:, -1]  # Last column (fairlead)

# Create dataset with lookback
def create_dataset(features, target, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback, :])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)

X, y = create_dataset(features, target, lookback)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)  # Dropout layer after LSTM
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])  # Apply dropout
        out = self.fc(x)
        return out

# Instantiate model
model = LSTM(input_size=6, hidden_size=64, output_size=1, num_layers=2).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
train_dataset = TensorDataset(X_train, y_train)
loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Training
n_epochs = 30
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
plt.plot(y_pred_inv[::4], label='Predicted Data (LSTM)', color='red')
plt.text(0.05, 0.95, f'RMSE: {rmse_value:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.show()
