import csv
import numpy as np
import matplotlib.pyplot as plt

with open("data.csv") as f:
    data = list(csv.reader(f))[1:]

X = np.array([float(r[0]) for r in data])
y = np.array([float(r[1]) for r in data])

print(data[:5])

n = int(0.8*len(X))

Xtr,Xt = X[:n],X[n:]
ytr,yt = y[:n],y[n:]

xm = np.mean(Xtr)
ym = np.mean(ytr)

theta1 = np.sum((Xtr-xm)*(ytr-ym)) / \
         np.sum((Xtr-xm)**2)

theta0 = ym-theta1*xm

print("Theta0 =",theta0)
print("Theta1 =",theta1)

pred = theta0+theta1*Xt

print("\nActual vs Predicted")
for a,p in zip(yt,pred):
    print(a,p)

for size in [1500,1700,2500]:
    print(size,"=",theta0+theta1*size)

mae = np.mean(np.abs(yt-pred))
mse = np.mean((yt-pred)**2)
rmse = np.sqrt(mse)

r2 = 1-np.sum((yt-pred)**2)/ \
       np.sum((yt-np.mean(yt))**2)

print("\nMAE =",mae)
print("MSE =",mse)
print("RMSE =",rmse)
print("R2 =",r2)

plt.scatter(X,y)
plt.plot(X,theta0+theta1*X)
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.show()

plt.scatter(yt,pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()