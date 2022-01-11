import json_lines
import spacy # fast NLP
import pandas as pd # dataframes
import langid # language identification (i.e. what language is this?)
from nltk.classify.textcat import TextCat # language identification from NLTK
from matplotlib.pyplot import plot # not as good as ggplot in R :p
from google_trans_new import google_translator
import csv
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

X= [ ] ; y = [ ] ; z =[]; lang = []
Xen = [];
Xother = [];
Xothercount = [];
with open ( 'reviews_125.jl' , 'rb' ) as f :
    for item in json_lines.reader(f):
        X.append ( str(item [ 'text' ]) )
        y.append ( item [ 'voted_up' ] )
        z.append ( item [ 'early_access'] )
        lang.append(langid.classify(item['text'])[0])

for index in range(len(lang)):
    if lang[index] !='en':
        if lang[index] not in Xother:
            Xother.append(lang[index])

            Xothercount.append(1)
        else:
            Xothercount[Xother.index(lang[index])]+=1 #Find other langs and their counts

THRESHOLD = 50
Xotherfiltered =[]
Xotherfilteredcount = []

for XotherInd in range(len(Xother)):
    if Xothercount[XotherInd] >= THRESHOLD:
        Xotherfiltered.append(Xother[XotherInd])
        Xotherfilteredcount.append(Xothercount[XotherInd])



Xfore= [ ] ; yfore = [ ] ; zfore =[]; langfore = []

translator = google_translator()


# writing to csv change false to be 0 true to be 1 and refill dataset

with open("reviews.csv", "w", newline="", encoding="utf-8") as f: # when/if google api works again can add foreign reviews also currently only have just over half of the dataset
    writer = csv.writer(f)
    for index in range(len(lang)):
        if lang[index] == 'en':
            writer.writerow([str(X[index]), 0 if str(y[index])=="False" else 1, 0 if str(z[index])=="False" else 1, lang[index]])
        if lang[index] in Xotherfiltered:
            print(translator.translate(X[index]))
            Xfore.append(translator.translate(X[index],lang_tgt=lang[index]) )
            yfore.append(y[index])
            zfore.append(z[index])
            langfore.append(lang[index])
            writer.writerow([translator.translate(X[index],lang_tgt=lang[index]), y[index], z[index], lang[index]])
            print(index)




dataset = pd.read_csv('reviews.csv', delimiter = ',', encoding='utf-8')

import re
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords') #unimportant words dataset (unimportant to machines atleast)
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, len(dataset)):

    #Remove anything that's not a-z or A-Z
    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i][0])#.decode('utf-8') if type(dataset.iloc[0,i]) is bytes else dataset.iloc[0,i]) # no 0/10 etc # cant do it on byte type or whatever

    review = review.lower() # standardising text and splitting it up/tokenising
    review = review.split()

    ps = PorterStemmer()
    #STEMMING
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Stringifying
    review = ' '.join(review)

    corpus.append(review) # Standardised text is appended to our corpus

#USING BoW/BAG OF WORDS MODEL

from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(max_features=2000)

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values # Upvoted or not


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # standard train/test split


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=700,criterion='entropy') # using information gain over # gini impurity

model.fit(X_train, y_train) # fitting model to split data

y_pred = model.predict(X_test) # prediction

print(y_pred)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc

confm = confusion_matrix(y_test, y_pred)
print(confm) #checking false positives/negatives

from sklearn.metrics import roc_curve
plt.rc('font', size=8); plt.rcParams['figure.constrained_layout.use'] = True
y_pred = [x[1] for x in model.predict_proba(X_test)]
fpr, tpr, _ = roc_curve(y_test,y_pred) #random forest
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,lw=2, label='Random Forest (estimators = 700) curve (area = %0.2f)' % roc_auc)


#LINEARSVC

from sklearn.svm import LinearSVC
model = LinearSVC(C=0.5).fit(X_train, y_train)

fpr, tpr, _ = roc_curve(y_test,model.decision_function(X_test)) #linearsvc
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,lw=2, label='LinearSVC (C=0.5) curve (area = %0.2f)' % roc_auc)


#KNN

KN = KNeighborsClassifier(n_neighbors=10, weights='uniform').fit(X_train, y_train)
y_pred = KN.predict(X_test)
y_pred = [x[1] for x in KN.predict_proba(X_test)]
fpr, tpr, _ = roc_curve(y_test,y_pred) #random forest
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,lw=2, label='KNeighbours(10) curve (area = %0.2f)' % roc_auc)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green',linestyle='--', label = "Baseline/Most common")
plt.legend(loc="lower right")
plt.show()



#EARLY ACCESS same procedure as before operating on same data (except using z column instead of y)


z = dataset.iloc[:, 2].values
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)


model = RandomForestClassifier(n_estimators=700, criterion='entropy')

model.fit(X_train, z_train)

z_pred = model.predict(X_test)

print(z_pred)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(z_test, z_pred)
print(cm)

from sklearn.metrics import roc_curve
plt.rc('font', size=8); plt.rcParams['figure.constrained_layout.use'] = True
z_pred = [x[1] for x in model.predict_proba(X_test)]

fpr, tpr, _ = roc_curve(z_test,z_pred) #random forest

roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,lw=2, label='Random Forest (estimators = 700) curve (area = %0.2f)' % roc_auc)





from sklearn.svm import LinearSVC
model = LinearSVC(C=0.5).fit(X_train, z_train)

fpr, tpr, _ = roc_curve(z_test,model.decision_function(X_test)) #linearsvc

roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,lw=2, label='LinearSVC (C=0.5) curve (area = %0.2f)' % roc_auc)


#KNN

KN = KNeighborsClassifier(n_neighbors=10, weights='uniform').fit(X_train, z_train)
z_pred = KN.predict(X_test)
z_pred = [x[1] for x in KN.predict_proba(X_test)]
fpr, tpr, _ = roc_curve(z_test,z_pred) #random forest
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,lw=2, label='KNeighbours(10) curve (area = %0.2f)' % roc_auc)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green',linestyle='--', label = "Baseline/Most common")
plt.legend(loc="lower right")
plt.show()

print(z_pred)
cm = confusion_matrix(z_test, z_pred)
print(cm)
