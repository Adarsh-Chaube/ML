import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from collections import Counter


'''Reading the csv file'''
df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\Email_spam_balanced_dataset.csv")

'''Creating Vocabulary'''
stop=set(stopwords.words('english'))
text=" ".join(df["Body"])
text=re.sub(r"[^a-zA-Z\s]",'',text.lower())
freq=Counter()

tokens=word_tokenize(text)
vocab=[word for word in tokens if word not in stop and len(word)>1 and len(word)<15]

'''Just keep the 10000 most frequent words'''
freq.update(vocab)
vocab=[word for word,count in freq.most_common(10000)]
print(len(vocab))
print(len(df["Body"]))

'''Saving Vocab as a pickle file'''
with open(r'C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\vocab.pkl','wb') as f:
    pickle.dump(vocab,f)