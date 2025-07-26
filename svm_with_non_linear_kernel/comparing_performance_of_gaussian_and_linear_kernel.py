import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from sklearn.model_selection import train_test_split

"""Load and preprocess the data"""
df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\dataset\kidney_disease_dataset\kidney_disease_dataset.csv")
df=df.to_numpy()
X=df[:,:7]
Y=df[:,7]
Y=[-1 if x==0 else x for x in Y]
Y=np.array(Y)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

"""split into train and test set"""
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=42)



'''Initialize a gaussian kernel'''
def gaussian_kernel(X,sigma=0.1):
    m=X.shape[0]
    k=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            d=X[i]-X[j]
            ed=np.dot(d,d)
            k[i,j]=np.exp(-ed/(2*sigma**2))
    return k


def pred_for_linear_kernel():
    """Linear Kernel"""
    k=np.dot(x_train,x_train.T)

    """Initializing Parameters for convex optimization"""
    """Regularization Parameter"""
    C=15

    """Initializing Quadratic programming matrices for soft margin svm"""
    m=x_train.shape[0]
    P=matrix(np.outer(y_train,y_train)*k.astype(np.double))
    q=matrix(-np.ones(m))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.vstack((np.zeros((m,1)), C * np.ones((m,1)))))
    A=matrix(y_train.astype(np.double),(1,m))
    b=matrix(0.0)

    """Solve optimization problem"""
    solvers.options['show_progress']=False
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
    print("Accuracy using linear kernel : ",accuracy)
  
  
  
  
    
def pred_for_gaussian_kernel():
    sigma=1.2
    k=gaussian_kernel(x_train,sigma)
    
    """Initializing Parameters for convex optimization"""
    """Regularization Parameter"""
    C=10

    """Initializing Quadratic programming matrices for soft margin svm"""
    m=x_train.shape[0]
    P=matrix(np.outer(y_train,y_train)*k.astype(np.double))
    q=matrix(-np.ones(m))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.vstack((np.zeros((m,1)), C * np.ones((m,1)))))
    A=matrix(y_train.astype(np.double),(1,m))
    b=matrix(0.0)

    """Solve optimization problem"""
    solvers.options['show_progress']=False
    solvers.options['maxiters'] = 20
    s=solvers.qp(P,q,G,h,A,b)
    alpha=np.ravel(s['x'])

    """Identify Support Vectors for creating decision boundary"""
    sv=alpha>1e-5
    svectors=x_train[sv]
    slabels=y_train[sv]
    salphas=alpha[sv]
    
    
    """Compute bias"""
    def compute_bias():
        bval=[]
        for i in range(len(salphas)):
            s=0
            for j in range(len(salphas)):
                s+=salphas[j]*slabels[j]*k[i,j]
            bi=slabels[i]-s
            bval.append(bi)
        return np.mean(bval)
    
    b=compute_bias()
    
    """function to map input to higher dimension"""
    def gaussian(x1,x2,sigma=0.1):
        diff=x1-x2
        return np.exp(-np.dot(diff,diff)/(2*sigma**2))
    
    """Predict"""
    def predict(x):
        yp=[]
        for xi in x:
            total=0
            for i in range(len(salphas)):
                total+=salphas[i]*slabels[i]*gaussian(xi,svectors[i],sigma)
            total+=b
            yp.append(np.sign(total))
        return np.array(yp)
    
    """Evaluate"""
    y_pred=predict(x_test)
    accuracy=np.mean(y_pred==y_test)*100
    print("Accuracy using gaussian kernel : ",accuracy)
    
    
pred_for_linear_kernel()
pred_for_gaussian_kernel()