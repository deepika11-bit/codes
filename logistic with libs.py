import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression()
model.fit(Xtr,ytr)

prob = model.predict_proba(Xt)[:,1]
pred = model.predict(Xt)

print("Probabilities:")
print(prob)

print("\nActual   :",list(yt))
print("Predicted:",list(pred))
print("Accuracy :",accuracy_score(yt,pred))

# Threshold analysis
for threshold in [0.3,0.5,0.7]:
    p = (prob>=threshold).astype(int)
    print("Threshold:",threshold,
          "Accuracy:",accuracy_score(yt,p))