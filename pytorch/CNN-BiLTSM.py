import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
cuda = torch.device('cuda')
lookback = 3
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
roll = datasets.roll.to_numpy()
pitch = datasets.pitch.to_numpy()
yaw = datasets.yaw.to_numpy()
fairlead = datasets.fairlead.to_numpy()
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
scaler4 = MinMaxScaler()
scaler5 = MinMaxScaler()
heave = np.squeeze(scaler1.fit_transform(heave.reshape(-1,1)))
surge = np.squeeze(scaler2.fit_transform(surge.reshape(-1,1)))
sway = np.squeeze(scaler3.fit_transform(sway.reshape(-1,1)))
fairlead = np.squeeze(scaler4.fit_transform(fairlead.reshape(-1,1)))
data_stack = np.stack((heave, surge, sway, fairlead), axis = 1)
print(data_stack)
