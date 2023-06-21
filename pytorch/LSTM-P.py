import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

lookback = 3
datasets = pd.read_excel("D:\pythonProject\lib1\dataTur2200.xlsx", header=0)
datasets.columns = [
    "surge",
    "sway",
    "heave",
    "roll",
    "pitch",
    "yaw",
    "fairlead",
]
heave = datasets.heave.to_numpy()
surge = datasets.surge.to_numpy()
sway = datasets.sway.to_numpy()
fairlead = datasets.fairlead.to_numpy()
print(heave.shape,type(heave))#(2201,)
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
heave = scaler1.fit_transform(heave.reshape(-1,1))
heave = np.squeeze(heave)
surge = np.squeeze(scaler1.fit_transform(surge.reshape(-1,1)))
sway = np.squeeze(scaler1.fit_transform(sway.reshape(-1,1)))
fairlead = scaler2.fit_transform(fairlead.reshape(-1,1))


def create_dataset(heave, surge, sway, fairlead, lookback):
    X, y = [], []
    temp = []
    for i in range(len(heave)):
        temp.append(np.array([heave[i], surge[i], sway[i]]))
    for i in range(len(heave) - lookback):
        X.append(temp[i : (i + lookback)])
        y.append(fairlead[i+lookback])
    return torch.Tensor(np.array(X)), torch.Tensor(np.array(y))

X, y = create_dataset(heave, surge, sway, fairlead, lookback)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)
print(X_train.shape, y_train.shape, len(heave))
# exit()


class Model_one(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=256, num_layers=1,batch_first = False)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.linear(x)
        return x


model = Model_one()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(
    data.TensorDataset(X_train, y_train), shuffle=True, batch_size=10
)

n_epochs = 40
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        # print(X_batch.shape,y_batch.shape)
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
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    print(type(model(X_train)))
    train_plot = scaler2.inverse_transform(model(X_train).detach().cpu().numpy())
    test_plot = scaler2.inverse_transform(model(X_test).detach().cpu().numpy())
    origin_plot = scaler2.inverse_transform(fairlead.reshape(-1,1))
    print(train_plot.shape, test_plot.shape)
    pred_plot = np.array(train_plot.tolist() + test_plot.tolist())
    # print(pred_plot.shape,origin_plot.shape)
plt.ylim(500000, 2500000)
plt.plot(pred_plot[300:1500])
plt.plot(origin_plot[300:1500])
plt.show()