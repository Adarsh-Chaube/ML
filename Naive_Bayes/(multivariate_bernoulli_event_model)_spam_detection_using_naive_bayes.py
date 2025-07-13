import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import train_test_split

with open(r'C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\vocab.pkl','rb') as f:
    vocab=list(pickle.load(f))

print(vocab[:100])
'''Word to index mapping'''
word_index={word:idx for idx,word in enumerate(vocab)}
print(len(word_index))

'''Load data'''
df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\Email_spam_balanced_dataset.csv")
text=df['Body']
labels=df["Label"].values

'''cleaning and vectorizing text'''
text=[re.sub(r"[^a-zA-Z\s]",'',txt.lower()) for txt in text]
text=[word_tokenize(txt) for txt in text]
vector=np.zeros((len(text),len(word_index)))
for i in range(0,len(text)):
    for word in text[i]:
        if word in word_index:
            vector[i][word_index[word]]=1;


'''train test split'''
x_train, x_test, y_train, y_test = train_test_split(vector, labels, test_size=0.2, random_state=42)


'''Initializing Parameters with Laplace Smoothing'''
fiy=np.mean(y_train)
fi1=((x_train[y_train==1]).sum(axis=0)+1)/(x_train[y_train==1].sum()+ x_train.shape[1])
fi0=((x_train[y_train==0]).sum(axis=0)+1)/(x_train[y_train==0].sum()+x_train.shape[1])

'''Another method To prevent log 0 , other than laplace smoothing'''
'''eps = 1e-10
fi1 = np.clip(fi1, eps, 1)
fi0 = np.clip(fi0, eps, 1)'''


'''Testing'''
def predict(fiy,fi1,fi0,vec):
    logfi1=np.log(fi1)
    logfi0=np.log(fi0)
    log_fiy1=np.log(fiy)
    log_fiy0=np.log(1-fiy)
    
    logp1=np.sum(vec*logfi1)+log_fiy1
    logp0=np.sum(vec*logfi0)+log_fiy0
    return 1 if logp1>=logp0 else 0

pred=[predict(fiy,fi1,fi0,vec) for vec in x_test]
accuracy=(pred==y_test).mean()*100
print("Accuracy : ",accuracy)