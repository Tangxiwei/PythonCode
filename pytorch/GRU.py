import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from pylab import *

lookback = 1
hidden_size = 256
cuda = torch.device('cuda')
datasets = pd.read_excel("D:\datasets\dataTur2200.xlsx", header=0)
datasets.columns = [
    "surge",
    "sway",
    "heave",
    "roll",
    "pitch",
    "yaw",
    "ft2",
]
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
heave = datasets.heave.to_numpy()
surge = datasets.surge.to_numpy()
sway = datasets.sway.to_numpy()
ft2 = datasets.ft2.to_numpy()
heave = np.squeeze(scaler1.fit_transform(heave.reshape(-1, 1)))
surge = np.squeeze(scaler1.fit_transform(surge.reshape(-1, 1)))
sway = np.squeeze(scaler1.fit_transform(sway.reshape(-1, 1)))
ft2 = scaler2.fit_transform(ft2.reshape(-1, 1))


def create_dataset(heave, surge, sway, ft2, lookback):
    X, y = [], []
    temp = []
    for i in range(len(heave)):
        temp.append(np.array([heave[i], surge[i], sway[i]]))
    for i in range(len(heave) - lookback):
        X.append(temp[i:(i + lookback)])
        y.append(ft2[i + lookback])
    return torch.Tensor(np.array(X)).to(cuda), torch.Tensor(np.array(y)).to(cuda)


X, y = create_dataset(heave, surge, sway, ft2, lookback)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)



class Module_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, num_layers=1)
        self.linear = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        input_seq, _ = self.gru(input_seq)
        input_seq = self.linear(input_seq)
        input_seq = input_seq[:, -1, :]
        input_seq = self.relu(input_seq)
        return input_seq


model = Module_GRU().to(cuda)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=10)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_train.cpu()))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_test.cpu()))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    train_plot = scaler2.inverse_transform(model(X_train).detach().cpu().numpy())
    test_plot = scaler2.inverse_transform(model(X_test).detach().cpu().numpy())
    ft2_plot = scaler2.inverse_transform(ft2)
    # # shift train predictions for plotting
    # train_plot = np.ones_like(ft2) * np.nan
    # # print(train_plot,train_plot.shape)
    # # y_pred = model(X_train)
    # # print(y_pred.shape)
    # # y_pred = y_pred[:, -1, :]
    # train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # # shift test predictions for plotting
    # test_plot = np.ones_like(ft2) * np.nan
    # test_plot[train_size+lookback:len(ft2)] = model(X_test)[:, -1, :]
# print(ft2_plot[100:700:2])
# print(train_plot[100:700:2])
ylim(1000000, 2000000)
plt.plot(ft2_plot[100:700], c='b')
plt.plot(train_plot[100:700], c='r')
# plt.plot(test_plot, c='g')
plt.show()
