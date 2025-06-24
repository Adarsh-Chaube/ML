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

x_mean = np.mean(x[: ,1], axis=0)
x_std = np.std(x[:,1], axis=0)
x[:,1] = (x[:,1] - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std


para=np.random.randn(12,1)
para[0]=1
lr=0.01
costh=[]
for j in range(10000):
  epcost=0
  dj=0
  for i in range(len(x)):
    te=x[i].reshape(1,-1)
    cp=np.dot(te,para)
    cost=(cp-y[i])**2
    dj+=(cp-y[i])*te.reshape(-1,1)
    epcost+=cost.item()
  para=para-lr*dj/len(x)
  avgcost=epcost/len(x)
  costh.append(avgcost)
  if(j%100==0):
    print("Iteration : ",j," Cost : ",avgcost)

plt.figure(figsize=(8, 6))
plt.plot(range(10000), costh, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Cost (Mean Squared Error)')
plt.title('Cost Function vs. Iterations')
plt.grid(True)
plt.show()