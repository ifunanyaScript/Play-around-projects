# 1. Define model
# 2. Construct loss and optimizer
# 3. Traininig loop
#   - forward pass: computing the prediction and loss
#   - backward pass: gradients
#   - update weights 
#   - loop till training is done.


import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0. Prepare the data 
df = datasets.load_breast_cancer()
X, y = df.data, df.target

num_samples, num_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. Build model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model =  LogisticRegression(num_features)

        
# 2. Loss and optimzer
learning_rate = 0.35
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. Training loop
epochs = 1000
for epoch in range(epochs):
    # forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # backward pass
    loss.backward()
    
    # update
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 100 == 0:
        
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
with torch.no_grad():
    prediction = model(X_test)
    prediction_cls = prediction.round()
    acc = (prediction_cls.eq(y_test).sum()/float(y_test.shape[0]))*100
    print(f'Accuracy: {round(acc.item())}')
    