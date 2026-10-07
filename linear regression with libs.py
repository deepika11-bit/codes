import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv("data.csv")

print(df.head())

X = df.iloc[:,0:1]
y = df.iloc[:,1]

Xtr,Xt,ytr,yt = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()
model.fit(Xtr,ytr)

print("Theta0 =",model.intercept_)
print("Theta1 =",model.coef_[0])

pred = model.predict(Xt)

print("\nActual vs Predicted")
print(pd.DataFrame({
    "Actual":yt,
    "Predicted":pred
}))

for size in [1500,1700,2500]:
    print(size,"=",model.predict([[size]])[0])

print("MAE =",mean_absolute_error(yt,pred))
print("MSE =",mean_squared_error(yt,pred))
print("RMSE =",np.sqrt(mean_squared_error(yt,pred)))
print("R2 =",r2_score(yt,pred))

plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.show()

plt.scatter(yt,pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()