import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(r'C:\Users\USER\Desktop\MyDatasets\wine_data.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples
        
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

samples = iter(dataloader)
features, targets = samples.next()

input_features = features.shape[1]
hidden_size = 40
num_classes = len(np.unique(targets))

class Model(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_features, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Define some training parameters
model = Model(input_features, hidden_size, num_classes)
learning_rate = 0.01
epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(dataloader)
for epoch in range(epochs):
    for i, (features, targets) in enumerate(dataloader):
        # forward pass
        prediction = model(features)
        loss = criterion(prediction, targets)
        
        # empty gradients and backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')





