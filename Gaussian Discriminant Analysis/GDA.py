import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

"""read data from csv file"""

df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\ML\Gaussian Discriminant Analysis\breast+cancer+wisconsin+diagnostic\wdbc.data",delimiter=',',header=None)

data=df.to_numpy()
y=data[:,1]
y=np.array([1 if c=='M' else 0 for c in y])
x=data[:,2:].astype(np.float64)

x = (x - x.mean(axis=0)) / x.std(axis=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

'''INITIALIZING PARAMETERS'''
phi=np.mean(y_train)
mu0=(x_train[y_train==0].mean(axis=0))
mu1=(x_train[y_train==1].mean(axis=0))
m=x_train.shape[0]
n=x_train.shape[1]
sigma=np.zeros((x_train.shape[1],x_train.shape[1]))

'''Computing the common covariance matrix'''
for i in range(0,m):
    xi=x_train[i,:]
    mu=mu1 if y_train[i]==1 else mu0
    sigma += np.outer(xi - mu, xi - mu)

sigma/=m

'''PREDICT'''
def predict(xi,mu0,mu1,phi,sigma):
    p0=(1/(np.power(np.pi,n/2)*np.sqrt(np.linalg.det(sigma))))*(-0.5)*np.exp((xi-mu0).T @(np.linalg.inv(sigma))@(xi-mu0))*(1-phi)
    p1=(1/(np.power(np.pi,n/2)*np.sqrt(np.linalg.det(sigma))))*(-0.5)*np.exp((xi-mu1).T @(np.linalg.inv(sigma))@(xi-mu1))*(phi)
    return 1 if p1>p0 else 0

y_pred=np.array([predict(xi,mu0,mu1,phi,sigma) for xi in x_test])
accuracy=(y_pred==y_test).mean()*100
print("Accuracy : ",accuracy)
    


'''VISUALIZE'''

'''Considering just first 2 features'''
xv=x[:,:2]
mu0v=mu0[:2]
mu1v=mu1[:2]
sigmav=sigma[:2,:2]

'''Separate Malignant and Benign Datapoints'''
mp=xv[y==1]
bp=xv[y==0]

'''Plot the data'''
plt.figure(figsize=(10,10))
plt.scatter(mp[:,0],mp[:,1],c='red',label="Malignant",alpha=0.5)
plt.scatter(bp[:,0],bp[:,1],c='blue',label="Benign",alpha=0.5)

'''Plot the Gaussian contours'''
# Draw Gaussian contour for each class
def draw_gaussian(mu, sigma, color):
    from matplotlib.patches import Ellipse
    from scipy.stats import multivariate_normal

    # Create a grid
    x, y = np.mgrid[mu[0]-3:mu[0]+3:.05, mu[1]-3:mu[1]+3:.05]
    pos = np.dstack((x, y))

    # Evaluate the Gaussian PDF
    rv = multivariate_normal(mu, sigma)
    plt.contour(x, y, rv.pdf(pos), colors=color, levels=5, alpha=0.7)

draw_gaussian(mu0v, sigmav, 'blue')
draw_gaussian(mu1v, sigmav, 'red')

plt.title("GDA on Breast Cancer Data (2 Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()