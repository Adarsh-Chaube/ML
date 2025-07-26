import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cvxopt import solvers, matrix

"""Load and Preprocess the Data"""
df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\dataset\breast+cancer+wisconsin+diagnostic\wdbc.data",header=None)
df=df.to_numpy()
labels=df[:,1]
y=np.array([1 if i=='M' else -1 for i in labels])
x=np.array(df[:,2:])

'''Train Test Split'''
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

'''no. of training examples'''
m=x_train.shape[0]

'''Linear Kernel (gram) matrix'''
k=np.dot(x_train,x_train.T)

"""Regularization Parameter"""
C=0.2

"""Initializing Quadratic programming matrices for soft margin svm"""
P=matrix(np.outer(y_train,y_train)*k.astype(np.double))
q=matrix(-np.ones(m))
G = matrix(np.vstack((-np.eye(m), np.eye(m))))
h = matrix(np.vstack((np.zeros((m,1)), C * np.ones((m,1)))))
A=matrix(y_train.astype(np.double),(1,m))
b=matrix(0.0)

"""Solve optimization problem"""
solvers.options['show_progress']=True
solvers.options['maxiters'] = 20
s=solvers.qp(P,q,G,h,A,b)
alpha=np.ravel(s['x'])

"""Identify Support Vectors for creating decision boundary"""
sv=alpha>1e-5
svectors=x_train[sv]
slabels=y_train[sv]
salphas=alpha[sv]

"""Compute weight vector"""
w=np.sum(salphas[:,None]*slabels[:,None]*svectors,axis=0)

"""Compute Bias"""
b = np.mean(slabels - np.dot(svectors, w))

"""Predict"""
def predict(x):
    return np.sign(np.dot(x,w)+b)

"""Evaluate"""
y_pred=predict(x_test)
accuracy=np.mean(y_pred==y_test)*100
print("Accuracy : ",accuracy)
