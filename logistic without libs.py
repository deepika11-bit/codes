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
iterations = 1000

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))

for i in range(iterations):

    p = sigmoid(Xtr@w+b)

    error = p-ytr

    dw = Xtr.T@error/len(Xtr)
    db = np.mean(error)

    w -= lr*dw
    b -= lr*db

p = sigmoid(Xt@w+b)

print("Probabilities:")
print(p)

pred = (p>=0.5).astype(int)

print("\nActual   :",yt)
print("Predicted:",pred)
print("Accuracy :",np.mean(pred==yt))

# Threshold analysis
for threshold in [0.3,0.5,0.7]:
    pred = (p>=threshold).astype(int)
    print("Threshold:",threshold,
          "Accuracy:",np.mean(pred==yt))