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
lookback = 2
hiiden_size = 256
batch_size = 10
datasets = pd.read_excel("D:\datasets\dataTur2200.xlsx", header=0)
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
fairlead = np.squeeze(scaler2.fit_transform(fairlead.reshape(-1,1)))

data_stack = np.stack((heave, surge, sway, fairlead), axis = 1)
print(data_stack.shape)
def create_dataset(data,lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: (i + lookback)])
        y.append(data[i + lookback, 3])  # Only use the first feature as the output
    np.array(y).reshape(-1,1)
    return torch.Tensor(np.array((X))).to(cuda), torch.Tensor(np.array(y)).to(cuda)

X, y = create_dataset(data_stack,lookback)
print(X.shape,y.shape)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class MyLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.input_size = input_size
        self.batch_size = batch_size
        # self.hidden_cell = (
        #     torch.randn(self.num_directions*self.num_layers, self.batch_size, self.hidden_size).to(cuda),
        #     torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(cuda)
        # )
        self.lstm = nn.LSTM(self.input_size , self.hidden_size ,self.num_layers, batch_first= True).to(cuda)
        self.fc = nn.Linear(self.hidden_size,1).to(cuda)
        self.relu = nn.ReLU().to(cuda)

    def forward(self,x):
        x , _ = self.lstm(x)
        # print(x.shape)
        x = self.fc(x)
        x = x [:,-1,:]
        #x = self.relu(x)
        return x
# loader = data.DataLoader(
#     data.TensorDataset(X_train, y_train), shuffle=True, batch_size=10
# )
model = MyLSTM(X.shape[2],hidden_size=256,num_layers=1).to(cuda)
loss_fn = nn.MSELoss().to(cuda)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.003)
n_epochs = 100
loss_data = []
for epoch in range(n_epochs):
    model.train()
    # for X_batch, y_batch in loader:
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i: i + batch_size]
        X_batch, y_batch = X_train[indices], y_train[indices]
        # print(X_batch.shape,y_batch.shape)
        y_pred = model(X_batch)
        # loss = loss_fn(y_pred[-1,:], y_batch[-1,:])
        loss = loss_fn(y_pred, y_batch.unsqueeze(1))
        loss_data.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_train.cpu().unsqueeze(1)))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_test.cpu().unsqueeze(1)))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f LOSS %.4f" % (epoch, train_rmse, test_rmse,loss))

with torch.no_grad():
    train_plot = scaler2.inverse_transform(model(X_train).detach().cpu().numpy())
    test_plot = scaler2.inverse_transform(model(X_test).detach().cpu().numpy())
    origin_plot = scaler2.inverse_transform(fairlead.reshape(-1,1))
    pred_plot = np.array(train_plot.tolist() + test_plot.tolist())

# disp_time_series = pd.read_excel("D:\datasets\data_ts.xlsx", header=0)
# disp = disp_time_series['ts2'].to_numpy()
# disp_second_order = disp**2+2*disp
# lower_bound = 4462*disp_second_order + 1210000
# upper_bound = 4462*disp_second_order + 1370000
# print(lower_bound[-20:], upper_bound[-20:],pred_plot[-20:])
# pred_plot = np.clip(pred_plot, lower_bound, upper_bound)

loss_data_list = []
for i in range(len(loss_data)):
    temp = loss_data[i].detach().cpu().numpy()
    loss_data_list.append(temp)
plt.ylim(1200000, 2000000)

# 添加标题和轴标签
plt.title('Original vs Predicted Data')
plt.xlabel('Time')
plt.ylabel('Fairlead Value')
plt.plot(origin_plot[1800:2200],label='Original Data', color='blue')
# pred_plot[1497:1500] = [[1586840.625],[1562812.25],[1456126.25]]
plt.plot(pred_plot[1800:2200],label='Predicted Data', color='red', linestyle='--')
# plt.ylim(0.00001,0.005)
# plt.plot(loss_data_list)
plt.legend()
plt.show()

