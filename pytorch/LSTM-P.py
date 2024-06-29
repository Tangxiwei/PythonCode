import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
cuda = torch.device('cuda')
lookback = 4
datasets = pd.read_excel("D:\datasets\dataTur2200.xlsx", header=0)
print(type(datasets))
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
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
heave = np.squeeze(scaler1.fit_transform(heave.reshape(-1,1)))
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
    return torch.Tensor(np.array(X)).to(cuda), torch.Tensor(np.array(y)).to(cuda)

X, y = create_dataset(heave, surge, sway, fairlead, lookback)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)
print(X_train.shape, y_train.shape, X_test.shape, len(heave))
# exit()

class Model_one(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=1024, num_layers=2,batch_first = True).to(cuda)
        self.relu = nn.ReLU().to(cuda)
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x[:, -1, :]
        # x = self.relu(x)
        return x

model = Model_one().to(cuda)
optimizer = optim.Adam(model.parameters(),lr = 0.01)
loss_fn = nn.MSELoss().to(cuda)
loader = data.DataLoader(
    data.TensorDataset(X_train, y_train), shuffle=True, batch_size=10
)

n_epochs = 10
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
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f LOSS %.4f" % (epoch, train_rmse, test_rmse,loss))

with torch.no_grad():
    print(type(model(X_train)))
    train_plot = scaler2.inverse_transform(model(X_train).detach().cpu().numpy())
    test_plot = scaler2.inverse_transform(model(X_test).detach().cpu().numpy())
    origin_plot = scaler2.inverse_transform(fairlead.reshape(-1,1))
    print(train_plot.shape, test_plot.shape)
    pred_plot = np.array(train_plot.tolist() + test_plot.tolist())
    # print(pred_plot.shape,origin_plot.shape)
# plt.ylim(500000, 2500000)
# plt.plot(pred_plot[300:1500])
# plt.plot(origin_plot[300:1500])
# plt.show()
loss_data_list = []
for i in range(len(loss_data)):
    temp = loss_data[i].detach().cpu().numpy()
    loss_data_list.append(temp)
plt.figure(figsize= (60,45))
ax1 = plt.subplot(2,1,1)
plt.ylim(500000, 2500000)
plt.plot(pred_plot[300:1500])
plt.plot(origin_plot[300:1500])
# ax2 = plt.subplot(2,1,1)
# plt.ylim(0.00001,0.005)
# plt.plot(loss_data_list)
# plt.show()