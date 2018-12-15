{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red26\green26\blue26;\red255\green255\blue255;\red62\green0\blue63;
}
{\*\expandedcolortbl;;\cssrgb\c13333\c13333\c13333;\cssrgb\c100000\c100000\c100000;\cssrgb\c31373\c0\c31373;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs26 \cf2 \cb3 \expnd0\expndtw0\kerning0
import os\cb1 \
\cb3 import glob\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 import re\cb1 \
\cb3 import numpy as np\cb1 \
\cb3 import pandas as pd\cb1 \
\cb3 import nltk\cb1 \
\cb3 import matplotlib.pyplot as plt\cb1 \
\cb3 \'a0\cb1 \
\cb3 #os.getcwd()\cb1 \
\
\cb3 """os.chdir('C:\\\\R\\\\Project\\\\sentiment labelled sentences')"""\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 path='C:\\\\R\\\\Project\\\\sentiment labelled sentences'\cb1 \
\
\cb3 all_Files=['C:\\\\R\\\\Project\\\\sentiment labelled sentences\\\\amazon_cells_labelled.txt',\cb1 \
\cb3 \'a0 \'a0 \'a0 \'a0 \'a0 \'a0'C:\\\\R\\\\Project\\\\sentiment labelled sentences\\\\imdb_labelled.txt']\cb1 \
\
\cb3 dfs=[]\cb1 \
\cb3 for f in all_Files:\cb1 \
\cb3 \'a0 \'a0 dfs.append(pd.read_csv(f, header=None, sep=r'\\t', engine='python'))\cb1 \
\cb3 \'a0 \'a0 """np_array_list.append(f.as_matrix())"""\cb1 \
\
\cb3 data=pd.concat(dfs, ignore_index=True)\cb1 \
\cb3 """comb_np_array=np.vstack(np_array_list)\cb1 \
\cb3 data=pd.DataFrame(comb_np_array)"""\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 """data=pd.read_csv(dataset, header=None, sep=r'\\t', engine='python')\cb1 \
\cb3 """\cb1 \
\cb3 data.columns=['review','sentiment']\cb1 \
\
\cb3 from sklearn.model_selection import train_test_split\cb1 \
\cb3 corpus, test_corpus, y, yt =train_test_split(data.iloc[:,0], data.iloc[:,1],test_size=0.25, random_state=101)\cb1 \
\
\cb3 from sklearn.feature_extraction import text\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 vectorizer=text.CountVectorizer(ngram_range=(1,2),stop_words='english').fit(corpus)\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 Tfidf=text.TfidfTransformer()\cb1 \
\cb3 X=Tfidf.fit_transform(vectorizer.transform(corpus))\cb1 \
\cb3 Xt=Tfidf.transform(vectorizer.transform(test_corpus))\cb1 \
\
\cb3 from sklearn.svm import LinearSVC\cb1 \
\cb3 from sklearn.grid_search import GridSearchCV\cb1 \
\cb3 param_grid=\{'C':[0.01, 0.1, 1.0, 10.0, 100.0]\}\cb1 \
\cb3 clf=GridSearchCV(LinearSVC(loss='hinge', random_state=101), param_grid)\cb1 \
\cb3 clf=clf.fit(X,y)\cb1 \
\cb3 print("Best Parameters: %s" % clf.best_params_)\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 from sklearn.metrics import accuracy_score\cb1 \
\cb3 solution=clf.predict(Xt)\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 print("Achieved accuracy \\nSVM:\{0:.1f\}%".format(accuracy_score(yt, solution)*100 ))\cb1 \
\
\
\cb3 #Accuracy vs C\cb1 \
\cb3 pgrid=[0.01, 0.1, 1.0, 10.0, 100.0]\cb1 \
\cb3 accur=[]\cb1 \
\cb3 for p in pgrid:\cb1 \
\cb3 \'a0 \'a0 param_grid['C']=[p]\cb1 \
\cb3 \'a0 \'a0 clfplot=GridSearchCV(LinearSVC(loss='hinge', random_state=101, penalty='l2'), param_grid)\cb1 \
\cb3 \'a0 \'a0 clfplot=clfplot.fit(X,y)\cb1 \
\cb3 \'a0 \'a0 solu=clfplot.predict(Xt)\cb1 \
\cb3 \'a0 \'a0 acc=accuracy_score(yt,solu)\cb1 \
\cb3 \'a0 \'a0 accur.append(acc*100)\cb1 \
\
\cb3 plt.plot(pgrid, accur)\cb1 \
\cb3 plt.ylabel('Accuracy (%)')\cb1 \
\cb3 plt.xlabel('C value')\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 #Accuracy vs penalty\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 lo=['hinge','squared_hinge']\cb1 \
\cb3 param_grid=\{'C':[0.01, 0.1, 1.0, 10.0, 100.0]\}\cb1 \
\cb3 accur=[]\cb1 \
\cb3 for l in lo:\cb1 \
\cb3 \'a0 \'a0 clfplot2=GridSearchCV(LinearSVC(loss=l, random_state=101, penalty='l2'), param_grid)\cb1 \
\cb3 \'a0 \'a0 clfplot2=clfplot2.fit(X,y)\cb1 \
\cb3 \'a0 \'a0 solu=clfplot2.predict(Xt)\cb1 \
\cb3 \'a0 \'a0 acc=accuracy_score(yt,solu)\cb1 \
\cb3 \'a0 \'a0 accur.append(acc*100)\cb1 \
\
\cb3 x=np.array([0,1])\cb1 \
\cb3 plt.ylabel('Accuracy (%)')\cb1 \
\cb3 plt.xticks(x,lo)\cb1 \
\cb3 plt.xlabel('Loss function')\cb1 \
\cb3 plt.plot(x, accur)\cb1 \
\
\
\
\cb3 #Accuracy vs ngram range\cb1 \
\cb3 from sklearn.svm import LinearSVC\cb1 \
\cb3 from sklearn.grid_search import GridSearchCV\cb1 \
\cb3 accur=[]\cb1 \
\cb3 grams=range(1,5)\cb1 \
\cb3 for r in grams:\cb1 \
\cb3 \'a0 \'a0 vectorizer=text.CountVectorizer(ngram_range=(r,r),stop_words='english').fit(corpus)\cb1 \
\cb3 \'a0 \'a0 Tfidf=text.TfidfTransformer()\cb1 \
\cb3 \'a0 \'a0 X=Tfidf.fit_transform(vectorizer.transform(corpus))\cb1 \
\cb3 \'a0 \'a0 Xt=Tfidf.transform(vectorizer.transform(test_corpus))\cb1 \
\cb3 \'a0 \'a0\'a0\cb1 \
\cb3 \'a0 \'a0 param_grid=\{'C':[0.01, 0.1, 1.0, 10.0, 100.0]\}\cb1 \
\cb3 \'a0 \'a0 clf3=GridSearchCV(LinearSVC(loss='hinge', random_state=101), param_grid)\cb1 \
\cb3 \'a0 \'a0 clf3=clf3.fit(X,y)\cb1 \
\cb3 \'a0 \'a0 solu=clf3.predict(Xt)\cb1 \
\cb3 \'a0 \'a0 acc=accuracy_score(yt,solu)\cb1 \
\cb3 \'a0 \'a0 accur.append(acc*100)\cb1 \
\
\cb3 plt.plot(grams, accur)\cb1 \
\cb3 plt.ylabel('Accuracy (%)')\cb1 \
\cb3 plt.xlabel('ngram')\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 #plt.show()\cb1 \
\
\cb3 #, accuracy_score(yt, solution2)*100)\'a0\cb1 \
\cb3 #print(test_corpus[yt!=solution], yt, solution)\cb1 \
\
\
\
\
\cb3 #print(test_corpus[yt!=solution], yt, solution)}