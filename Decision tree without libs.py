import csv
import math
import random
import time
with open("data.csv") as f:
    data = list(csv.reader(f))[1:]
random.shuffle(data)
n = int(0.8 * len(data))
train, test = data[:n], data[n:]
X = [r[:-1] for r in train]
y = [r[-1] for r in train]
Xt = [r[:-1] for r in test]
yt = [r[-1] for r in test]
def counts(y):
    d = {}
    for v in y:
        d[v] = d.get(v, 0) + 1
    return d
def entropy(y):
    n = len(y)
    return -sum((c/n)*math.log2(c/n) for c in counts(y).values())
def gini(y):
    n = len(y)
    return 1-sum((c/n)**2 for c in counts(y).values())
def split(X,y,j):
    d = {}
    for x,t in zip(X,y):
        d.setdefault(x[j],[[],[]])
        d[x[j]][0].append(x)
        d[x[j]][1].append(t)
    return d
def gain(X,y,j):
    s = 0
    for a,b in split(X,y,j).values():
        s += len(b)/len(y)*entropy(b)
    return entropy(y)-s
def gain_ratio(X,y,j):
    d = split(X,y,j)
    si = 0
    for a,b in d.values():
        p = len(b)/len(y)
        si -= p*math.log2(p)
    return gain(X,y,j)/si if si else 0
def gini_index(X,y,j):
    return sum(len(b)/len(y)*gini(b)
               for a,b in split(X,y,j).values())
def build(X,y,attrs,method):
    if len(set(y)) == 1:
        return ("leaf",y[0])
    if not attrs:
        return ("leaf",max(counts(y),key=counts(y).get))
    if method == "ID3":
        j = max(attrs,key=lambda a:gain(X,y,a))
    elif method == "C4.5":
        j = max(attrs,key=lambda a:gain_ratio(X,y,a))
    else:
        j = min(attrs,key=lambda a:gini_index(X,y,a))
    tree = ("node",j,{})
    for v,(xx,yy) in split(X,y,j).items():
        tree[2][v] = build(
            xx,yy,[a for a in attrs if a != j],method
        )
    return tree
def predict(tree,x):
    if tree[0] == "leaf":
        return tree[1]
    j = tree[1]
    if x[j] not in tree[2]:
        return None
    return predict(tree[2][x[j]],x)
def metrics(a,p):
    labels = list(set(a))
    pos = labels[0]
    tp = sum(x==pos and z==pos for x,z in zip(a,p))
    fp = sum(x!=pos and z==pos for x,z in zip(a,p))
    fn = sum(x==pos and z!=pos for x,z in zip(a,p))
    acc = sum(x==z for x,z in zip(a,p))/len(a)
    pre = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*pre*rec/(pre+rec) if pre+rec else 0
    return acc,pre,rec,f1
def depth(t):
    if t[0] == "leaf":
        return 0
    return 1+max(depth(c) for c in t[2].values())
def leaves(t):
    if t[0] == "leaf":
        return 1
    return sum(leaves(c) for c in t[2].values())
for method in ["ID3","C4.5","CART"]:
    start = time.time()
    tree = build(
        X,y,list(range(len(X[0]))),method
    )
    tm = time.time()-start
    pred = [predict(tree,x) for x in Xt]
    acc,pre,rec,f1 = metrics(yt,pred)
    print("\n",method)
    print("Accuracy :",acc)
    print("Precision:",pre)
    print("Recall   :",rec)
    print("F1 Score :",f1)
    print("Tree Depth:",depth(tree))
    print("Leaf Nodes:",leaves(tree))
    print("Training Time:",tm)