import numpy as np
import matplotlib.pyplot as plt

d=[]
with open("C:\\Users\\lenovo\\Desktop\\ML\\housing price prediction\\Housing.csv","r") as file:
  d=file.readlines()
d=d[1:]
x=[]
y=[]
for line in d:
  l=line.strip().split(',');
  price=l[0];
  y.append(price);
  area=int(l[1]);
  bedrooms=int(l[2]);
  bathrooms=int(l[3]);
  stories=int(l[4]);
  mainroad= 1 if l[5]=="yes" else 0;
  guestroom=1 if l[6]=="yes" else 0;
  basement=1 if l[7]=="yes" else 0;
  hotwaterheating=1 if l[8]=="yes" else 0;
  airconditioning=1 if l[9]=="yes" else 0;
  parking=int(l[10]);
  furnishingstatus=1 if l[12]=="furnished" else 0;
  furnishingstatus=0.5 if l[12]=="semi-furnished" else furnishingstatus;
  x.append([1,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,furnishingstatus]);

x=np.array(x,dtype=float);
y=np.array(y,dtype=float);

x[:,1:] = (x[:,1:] - np.mean(x[:,1:], axis=0)) / np.std(x[:,1:], axis=0)

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std


para=np.dot(np.linalg.pinv(np.dot(x.T,x)),np.dot(x.T,y))

pred=np.dot(x,para)
cost=np.mean((pred-y)**2)
print("Cost : ",cost)