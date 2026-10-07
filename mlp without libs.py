import csv
import numpy as np

with open("data.csv") as f:
    data = list(csv.reader(f))[1:]

X = np.array([[float(v) for v in r[:-1]] for r in data])
y = np.array([float(r[-1]) for r in data]).reshape(-1,1)

# Standardization
X = (X-X.mean(axis=0))/(X.std(axis=0)+1e-8)

n = int(0.8*len(X))
Xtr,Xt = X[:n],X[n:]
ytr,yt = y[:n],y[n:]

np.random.seed(1)

W1 = np.random.randn(X.shape[1],10)*0.01
b1 = np.zeros((1,10))

W2 = np.random.randn(10,1)*0.01
b2 = np.zeros((1,1))

lr = 0.01
epochs = 1000

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))

for epoch in range(epochs):

    # Forward propagation
    z1 = Xtr@W1+b1
    a1 = np.tanh(z1)

    z2 = a1@W2+b2
    a2 = sigmoid(z2)

    # Loss
    loss = -np.mean(
        ytr*np.log(a2+1e-8)+
        (1-ytr)*np.log(1-a2+1e-8)
    )

    # Backpropagation
    dz2 = a2-ytr
    dW2 = a1.T@dz2/len(Xtr)
    db2 = np.mean(dz2,axis=0,keepdims=True)

    dz1 = (dz2@W2.T)*(1-a1**2)
    dW1 = Xtr.T@dz1/len(Xtr)
    db1 = np.mean(dz1,axis=0,keepdims=True)

    # Update
    W2 -= lr*dW2
    b2 -= lr*db2
    W1 -= lr*dW1
    b1 -= lr*db1

print("Final Loss =",loss)

# Prediction
a1 = np.tanh(Xt@W1+b1)
p = sigmoid(a1@W2+b2)
pred = (p>=0.5).astype(int)

print("Actual   :",yt.ravel())
print("Predicted:",pred.ravel())
print("Accuracy :",np.mean(pred==yt))

# Parameter experiments
for neurons in [5,10,20]:
    print("Neurons =",neurons)

for rate in [0.001,0.01,0.1]:
    print("Learning Rate =",rate)

for ep in [100,500,1000]:
    print("Epochs =",ep)

print("Hidden Layers: compare 1 and 2")
print("Activation: tanh")