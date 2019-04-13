#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
 
path='C:\\R\\Project\\sentiment labelled sentences'

all_Files=['C:\\R\\Project\\sentiment labelled sentences\\amazon_cells_labelled.txt',
           'C:\\R\\Project\\sentiment labelled sentences\\imdb_labelled.txt']

dfs=[]
for f in all_Files:
    dfs.append(pd.read_csv(f, header=None, sep=r'\t', engine='python'))

data_model=pd.concat(dfs, ignore_index=True)
data_model.columns=['review','sentiment']

data_model2=data_model.copy()

def negate_sequence(text):
    negation = False
    delims = "?.,!:;"
    result = ''
    words = text.split()
    for word in text.split():
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result=result+(negated)+' '

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False
    return result


for i in data_model.index:
    data_model['review'][i]=negate_sequence(data_model['review'][i])

#print(data_model['review'][3])
#data_model.head(100)

data=data_model.copy()

from sklearn.model_selection import train_test_split
corpus, test_corpus, y, yt =train_test_split(data.iloc[:,0], data.iloc[:,1],test_size=0.25, random_state=101)

from sklearn.feature_extraction import text
vectorizer=text.CountVectorizer(ngram_range=(1,1)).fit(corpus)
Tfidf=text.TfidfTransformer()
X=Tfidf.fit_transform(vectorizer.transform(corpus))
Xt=Tfidf.transform(vectorizer.transform(test_corpus))

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
#param_grid={'C':[0.01, 0.1, 1.0, 10.0, 100.0]}
#clf=GridSearchCV(LinearSVC(loss='hinge', random_state=101), param_grid)
clf=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=101, tol=0.0001, verbose=0)
clf=clf.fit(X,y)
#print("Best Parameters: %s" % clf.best_params_)

from sklearn.metrics import accuracy_score
solution=clf.predict(Xt)
print("Achieved accuracy \nSVM:{0:.1f}%".format(accuracy_score(yt, solution)*100 ))

#Extracting features
coef=clf.coef_.ravel()
#pos_coef=clf.coef_[2]
top_pos=np.argsort(coef)[-15:]
top_neg=np.argsort(coef)[:15]

top_coef=np.hstack([top_neg,top_pos])

feature_names=np.array(vectorizer.get_feature_names())
print(data[124:125])
for tc,c in zip(top_coef, coef):
    print(feature_names[tc],c)

#Testing
docs_new = ['not bad','how terrible']
X_new_tfidf = Tfidf.transform(vectorizer.transform(docs_new))

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))

#Enron dataset
os.chdir('C:\\R\\Project\\dtoEmails_TextOnly')
path2='C:\\R\\Project\\dtoEmails_TextOnly'
all_Files= glob.glob(os.path.join(path2,"*.txt"))
dfs=[]
for f in all_Files:
    dfs.append(pd.read_csv(f, header=None, sep=r'\n', engine='python'))

data=pd.concat(dfs, ignore_index=True)

todrop=[]

dropcond=['Date:','From:','To:','Subject:','Mime-Version:','cc:','bcc:','folder:',
          'origin:',"Content-Type:",'Filename:',"Content-Transfer-Encoding:","X-To:",
          "X-cc:","X-bcc:","X-Folder:","X-Origin:","X-FileName:",">From:",">Subject:",
          ">To:",">X-MIMETrack:","Bcc:","Cc:","Time:","@enron.com",
          "##########################################################"]

for i in data.index:
    for item in dropcond:
        if item in data[0][i]:
            todrop.append(i)

len(todrop)
data.head(70)
docids=[]
segids=[]

for i in data.index:
    if 'segmentNumber' in data[0][i]:
        segids.append([int(s) for s in data[0][i].split() if s.isdigit()])
        todrop.append(i)
    if 'docID' in data[0][i]:
        docids.append([int(s) for s in data[0][i].split() if s.isdigit()])
        todrop.append(i)

review=[]
reviewnew=''

len(data)

for i in data.index:
    if 'Body:' in data[0][i]:
        review.append(reviewnew)
        reviewnew=''
        data[0][i]=data[0][i].replace("Body:","")
        reviewnew=str(data[0][i])
    elif 'Body:\t' in data[0][i]:
        review.append(reviewnew)
        reviewnew=''
        data[0][i]=data[0][i].replace("Body:\t","")
        reviewnew=str(data[0][i])        
    else:
        reviewnew+=data[0][i]

len(review)    
len(docids)
len(segids)

docs_new = review.copy()
X_new_tfidf = Tfidf.transform(vectorizer.transform(docs_new))

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))