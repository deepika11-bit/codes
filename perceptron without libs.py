import csv
import numpy as np

with open("data.csv") as f:
    data = list(csv.reader(f))[1:]

X = np.array([[float(v) for v in r[:-1]] for r in data])
y = np.array([float(r[-1]) for r in data])

u = np.unique(y)
y = np.where(y==u[0],0,1)

# Scaling
X = (X-X.mean(0))/(X.std(0)+1e-8)

n = int(0.8*len(X))
Xtr,Xt = X[:n],X[n:]
ytr,yt = y[:n],y[n:]

w = np.zeros(X.shape[1])
b = 0
lr = 0.01
epochs = 100

def step(z):
    return 1 if z >= 0 else 0

for epoch in range(epochs):
    for x,t in zip(Xtr,ytr):

        z = np.dot(x,w)+b
        p = step(z)

        w += lr*(t-p)*x
        b += lr*(t-p)

pred = np.array([
    step(np.dot(x,w)+b) for x in Xt
])

print("Actual   :",yt)
print("Predicted:",pred)
print("Accuracy :",np.mean(pred==yt))

# Learning-rate analysis
for lr in [0.001,0.01,0.1]:

    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(100):
        for x,t in zip(Xtr,ytr):
            p = step(np.dot(x,w)+b)
            w += lr*(t-p)*x
            b += lr*(t-p)

    p = np.array([
        step(np.dot(x,w)+b) for x in Xt
    ])

    print("Learning Rate:",lr,
          "Accuracy:",np.mean(p==yt))

# Epoch analysis
for ep in [10,50,100]:
    print("Epochs:",ep)