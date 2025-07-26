import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

"""Load and preprocess the data"""
df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\dataset\kidney_disease_dataset\kidney_disease_dataset.csv")
X=df.iloc[:,0:7]
Y=df.iloc[:,7]
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


"""Performing EDA"""
eda_df = X.copy()
eda_df['Target'] = Y

"""Pairplot"""
'''sns.pairplot(eda_df, hue='Target')
plt.show()'''

"""Boxplot"""
'''for col in X.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=eda_df, x='Target', y=col)
    plt.title(f'Distribution of {col} by Target')
    plt.tight_layout()
    plt.show()'''

"""Correlation Heatmap"""
'''plt.figure(figsize=(10, 8))
sns.heatmap(eda_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()'''

selected_columns=[1,2,5]
Xs=X.iloc[:100,selected_columns]
Ys=Y.iloc[:100]
"""Plotting most prominent 3 features"""

# Separate points by class
X_pos = Xs[Ys == 1]
X_neg = Xs[Ys == 0]

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pos.iloc[:, 0], X_pos.iloc[:, 1], X_pos.iloc[:, 2], color='blue', label='Class 1', alpha=0.7)
ax.scatter(X_neg.iloc[:, 0], X_neg.iloc[:, 1], X_neg.iloc[:, 2], color='red', label='Class -1', alpha=0.7)

ax.set_xlabel('Creatinine_Level')
ax.set_ylabel('BUN')
ax.set_zlabel('GFR')
ax.legend()
plt.title('3D Plot of Three Features by Class')
plt.show()