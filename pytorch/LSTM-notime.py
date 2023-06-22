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
roll = datasets.roll.to_numpy()
pitch = datasets.pitch.to_numpy()
yaw = datasets.yaw.to_numpy()
fairlead = datasets.fairlead.to_numpy()
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
heave = np.squeeze(scaler1.fit_transform(heave.reshape(-1,1)))
surge = np.squeeze(scaler1.fit_transform(surge.reshape(-1,1)))
sway = np.squeeze(scaler1.fit_transform(sway.reshape(-1,1)))
fairlead = scaler2.fit_transform(fairlead.reshape(-1,1))
def create_datasets(heave,surge,sway,fairlead):
    X , y = [],[]
    for i in range(len(heave) ):
        X.append(np.array([heave[i],surge[i],sway[i]]))
        y = fairlead
    return torch.Tensor(np.array(X)).to(cuda), torch.Tensor(np.array(y)).to(cuda)
X,y = create_datasets(heave,surge,sway,fairlead)
print(X.shape,y.shape)#(2201,3).(2201,)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)
class LSTM_notime(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size= 3, hidden_size=256, num_layers=1,batch_first = True)
        self.linear = nn.Linear(256, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        #print(x.shape)[10,3]
        x, _ = self.lstm(x)
        # print(x.shape)[10,256]
        x = self.linear(x)
        x = self.relu(x)
        # print(x.shape)[10,1]
        return x

model =LSTM_notime().to(cuda)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True,batch_size = 10)

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
    origin_plot = scaler2.inverse_transform(fairlead)
    print(train_plot.shape,test_plot.shape)
    pred_plot = np.array(train_plot.tolist() + test_plot.tolist())
    # print(pred_plot.shape,origin_plot.shape)
plt.ylim(500000,2500000)
plt.plot(pred_plot[300:800])
plt.plot(origin_plot[690:1190])
plt.show()


