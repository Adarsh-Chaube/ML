import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



data=np.genfromtxt("C:\\Users\\lenovo\\Desktop\\ML\\Logistic_regression\\breast+cancer+wisconsin+diagnostic\\wdbc.data",delimiter=',')

X=np.array([row[2:-1] for row in data],dtype=np.float32)

Y=np.array([1 if row[1]=='M' else 0 for row in data],dtype=np.float32).reshape(-1,1)

def sigmoid(x):
    return (1.0/(1.0+np.exp(-x)))


X = (X - X.mean(axis=0)) / X.std(axis=0)

bias = np.ones((X.shape[0], 1))
X = np.append(bias, X, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
para=np.random.randn(X_train.shape[1],1)

iterations=100
lr=0.001
ll=[]
for i in range(iterations):
    z=np.dot(X_train,para)
    y_pred=sigmoid(z)
    eps=1e-8
    log_likelihood=np.sum(Y_train*np.log(y_pred+eps) + (1-Y_train)*np.log(1-y_pred+eps))
    ll.append(log_likelihood)
    para+=lr*np.dot(X_train.T,(Y_train-y_pred).reshape(-1,1))

y_pred_labels = (sigmoid(np.dot(X_test, para)) >= 0.5).astype(int)
accuracy = (y_pred_labels == Y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize=(8, 6))
plt.plot(range(iterations), ll, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Log likelihood')
plt.title('Log Likelihood vs. Iterations')
plt.grid(True)
plt.show()
        