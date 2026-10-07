import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
df = pd.read_csv("data.csv")
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
Xtr,Xt,ytr,yt = train_test_split(
    X,y,test_size=0.2,random_state=42
)
# ID3
start = time.time()
id3 = DecisionTreeClassifier(
    criterion="entropy",random_state=42
)
id3.fit(Xtr,ytr)
p1 = id3.predict(Xt)
t1 = time.time()-start
# CART
start = time.time()
cart = DecisionTreeClassifier(
    criterion="gini",random_state=42
)
cart.fit(Xtr,ytr)
p2 = cart.predict(Xt)
t2 = time.time()-start
def show(name,model,p,t):
    print("\n",name)
    print("Accuracy :",accuracy_score(yt,p))
    print("Precision:",precision_score(yt,p,average="weighted"))
    print("Recall   :",recall_score(yt,p,average="weighted"))
    print("F1 Score :",f1_score(yt,p,average="weighted"))
    print("Tree Depth:",model.get_depth())
    print("Leaf Nodes:",model.get_n_leaves())
    print("Training Time:",t)
show("ID3",id3,p1,t1)
show("CART",cart,p2,t2)
# C4.5 uses Gain Ratio.
# It is implemented normally because sklearn does not provide
# Gain Ratio as a DecisionTreeClassifier criterion.
print("\nC4.5: Use the Gain-Ratio implementation from Program 1.")