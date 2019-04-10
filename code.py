{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf400
{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red26\green26\blue26;\red255\green255\blue255;\red16\green60\blue192;
}
{\*\expandedcolortbl;;\cssrgb\c13333\c13333\c13333;\cssrgb\c100000\c100000\c100000;\cssrgb\c6667\c33333\c80000;
}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs26 \cf2 \cb3 \expnd0\expndtw0\kerning0
import os\cb1 \
\cb3 import glob\cb1 \
\cb3 import re\cb1 \
\cb3 import numpy as np\cb1 \
\cb3 import pandas as pd\cb1 \
\cb3 import nltk\cb1 \
\cb3 import matplotlib.pyplot as plt\cb1 \
\cb3 \'a0\cb1 \
\cb3 path='C:\\\\R\\\\Project\\\\sentiment labelled sentences'\cb1 \
\
\cb3 all_Files=['C:\\\\R\\\\Project\\\\sentiment labelled sentences\\\\amazon_cells_labelled.txt',\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 \'a0'C:\\\\R\\\\Project\\\\sentiment labelled sentences\\\\imdb_labelled.txt']\cb1 \
\
\cb3 dfs=[]\cb1 \
\cb3 for f in all_Files:\cb1 \
\cb3 \'a0 \'a0 dfs.append(pd.read_csv(f, header=None, sep=r'\\t', engine='python'))\cb1 \
\
\cb3 data_model=pd.concat(dfs, ignore_index=True)\cb1 \
\cb3 data_model.columns=['review','sentiment']\cb1 \
\
\cb3 data_model2=data_model.copy()\cb1 \
\
\cb3 def negate_sequence(text):\cb1 \
\cb3 \'a0 \'a0 negation = False\cb1 \
\cb3 \'a0 \'a0 delims = "?.,!:;"\cb1 \
\cb3 \'a0 \'a0 result = ''\cb1 \
\cb3 \'a0 \'a0 words = text.split()\cb1 \
\cb3 \'a0 \'a0 for word in text.split():\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 stripped = word.strip(delims).lower()\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 negated = "not_" + stripped if negation else stripped\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 result=result+(negated)+' '\cb1 \
\
\cb3 \'a0 \'a0 \'a0 \'a0 if any(neg in word for neg in ["not", "n't", "no"]):\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 \'a0 negation = not negation\cb1 \
\
\cb3 \'a0 \'a0 \'a0 \'a0 if any(c in word for c in delims):\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 \'a0 negation = False\cb1 \
\cb3 \'a0 \'a0 return result\cb1 \
\
\
\cb3 for i in data_model.index:\cb1 \
\cb3 \'a0 \'a0 data_model['review'][i]=negate_sequence(data_model['review'][i])\cb1 \
\
\cb3 #print(data_model['review'][3])\cb1 \
\cb3 #data_model.head(100)\cb1 \
\
\cb3 data=data_model.copy()\cb1 \
\
\cb3 from sklearn.model_selection import train_test_split\cb1 \
\cb3 corpus, test_corpus, y, yt =train_test_split(data.iloc[:,0], data.iloc[:,1],test_size=0.25, random_state=101)\cb1 \
\
\cb3 from sklearn.feature_extraction import text\cb1 \
\cb3 vectorizer=text.CountVectorizer(ngram_range=(1,1)).fit(corpus)\cb1 \
\cb3 Tfidf=text.TfidfTransformer()\cb1 \
\cb3 X=Tfidf.fit_transform(vectorizer.transform(corpus))\cb1 \
\cb3 Xt=Tfidf.transform(vectorizer.transform(test_corpus))\cb1 \
\
\cb3 from sklearn.svm import LinearSVC\cb1 \
\cb3 from sklearn.grid_search import GridSearchCV\cb1 \
\cb3 #param_grid=\{'C':[0.01, 0.1, 1.0, 10.0, 100.0]\}\cb1 \
\cb3 #clf=GridSearchCV(LinearSVC(loss='hinge', random_state=101), param_grid)\cb1 \
\cb3 clf=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,\cb1 \
\cb3 \'a0 \'a0 \'a0intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',\cb1 \
\cb3 \'a0 \'a0 \'a0penalty='l2', random_state=101, tol=0.0001, verbose=0)\cb1 \
\cb3 clf=clf.fit(X,y)\cb1 \
\cb3 #print("Best Parameters: %s" % clf.best_params_)\cb1 \
\
\cb3 from sklearn.metrics import accuracy_score\cb1 \
\cb3 solution=clf.predict(Xt)\cb1 \
\cb3 print("Achieved accuracy \\nSVM:\{0:.1f\}%".format(accuracy_score(yt, solution)*100 ))\cb1 \
\
\cb3 #Extracting features\cb1 \
\cb3 coef=clf.coef_.ravel()\cb1 \
\cb3 #pos_coef=clf.coef_[2]\cb1 \
\cb3 top_pos=np.argsort(coef)[-15:]\cb1 \
\cb3 top_neg=np.argsort(coef)[:15]\cb1 \
\
\cb3 top_coef=np.hstack([top_neg,top_pos])\cb1 \
\
\cb3 feature_names=np.array(vectorizer.get_feature_names())\cb1 \
\cb3 print(data[124:125])\cb1 \
\cb3 for tc,c in zip(top_coef, coef):\cb1 \
\cb3 \'a0 \'a0 print(feature_names[tc],c)\cb1 \
\
\cb3 #Testing\cb1 \
\cb3 docs_new = ['not bad','how terrible']\cb1 \
\cb3 X_new_tfidf = Tfidf.transform(vectorizer.transform(docs_new))\cb1 \
\
\cb3 predicted = clf.predict(X_new_tfidf)\cb1 \
\cb3 for doc, category in zip(docs_new, predicted):\cb1 \
\cb3 \'a0 \'a0 print('%r => %s' % (doc, category))\cb1 \
\
\cb3 #Enron dataset\cb1 \
\cb3 os.chdir('C:\\\\R\\\\Project\\\\dtoEmails_TextOnly')\cb1 \
\cb3 path2='C:\\\\R\\\\Project\\\\dtoEmails_TextOnly'\cb1 \
\cb3 all_Files= glob.glob(os.path.join(path2,"*.txt"))\cb1 \
\cb3 dfs=[]\cb1 \
\cb3 for f in all_Files:\cb1 \
\cb3 \'a0 \'a0 dfs.append(pd.read_csv(f, header=None, sep=r'\\n', engine='python'))\cb1 \
\
\cb3 data=pd.concat(dfs, ignore_index=True)\cb1 \
\
\cb3 todrop=[]\cb1 \
\
\cb3 dropcond=['Date:','From:','To:','Subject:','Mime-Version:','cc:','bcc:','folder:',\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 'origin:',"Content-Type:",'Filename:',"Content-Transfer-Encoding:","X-To:",\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 "X-cc:","X-bcc:","X-Folder:","X-Origin:","X-FileName:",">From:",">Subject:",\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 ">To:",">X-MIMETrack:","Bcc:","Cc:","Time:","@{\field{\*\fldinst{HYPERLINK "http://enron.com/"}}{\fldrslt \cf4 \ul \ulc4 enron.com}}",\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 "##########################################################"]\cb1 \
\
\cb3 for i in data.index:\cb1 \
\cb3 \'a0 \'a0 for item in dropcond:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 if item in data[0][i]:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 \'a0 todrop.append(i)\cb1 \
\
\cb3 len(todrop)\cb1 \
\cb3 data.head(70)\cb1 \
\cb3 docids=[]\cb1 \
\cb3 segids=[]\cb1 \
\
\cb3 for i in data.index:\cb1 \
\cb3 \'a0 \'a0 if 'segmentNumber' in data[0][i]:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 segids.append([int(s) for s in data[0][i].split() if s.isdigit()])\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 todrop.append(i)\cb1 \
\cb3 \'a0 \'a0 if 'docID' in data[0][i]:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 docids.append([int(s) for s in data[0][i].split() if s.isdigit()])\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 todrop.append(i)\cb1 \
\
\cb3 review=[]\cb1 \
\cb3 reviewnew=''\cb1 \
\
\cb3 len(data)\cb1 \
\
\cb3 for i in data.index:\cb1 \
\cb3 \'a0 \'a0 if 'Body:' in data[0][i]:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 review.append(reviewnew)\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 reviewnew=''\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 data[0][i]=data[0][i].replace("Body:","")\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 reviewnew=str(data[0][i])\cb1 \
\cb3 \'a0 \'a0 elif 'Body:\\t' in data[0][i]:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 review.append(reviewnew)\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 reviewnew=''\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 data[0][i]=data[0][i].replace("Body:\\t","")\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 reviewnew=str(data[0][i])\'a0 \'a0 \'a0 \'a0\'a0\cb1 \
\cb3 \'a0 \'a0 else:\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 reviewnew+=data[0][i]\cb1 \
\
\cb3 len(review)\'a0 \'a0\'a0\cb1 \
\cb3 len(docids)\cb1 \
\cb3 len(segids)\cb1 \
\
\cb3 docs_new = review.copy()\cb1 \
\cb3 X_new_tfidf = Tfidf.transform(vectorizer.transform(docs_new))\cb1 \
\
\cb3 predicted = clf.predict(X_new_tfidf)\cb1 \
\cb3 for doc, category in zip(docs_new, predicted):\cb1 \
\cb3 \'a0 \'a0 print('%r => %s' % (doc, category))}