import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

Xtr,Xt,ytr,yt = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

sc = StandardScaler()

Xtr = sc.fit_transform(Xtr)
Xt = sc.transform(Xt)

for k in [3,5,7]:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(Xtr,ytr)

    pred = model.predict(Xt)

    print("K =",k)
    print("Actual   :",list(yt))
    print("Predicted:",list(pred))
    print("Accuracy :",accuracy_score(yt,pred))