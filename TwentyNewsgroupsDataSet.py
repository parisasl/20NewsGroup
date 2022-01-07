# Objective:

# The 20 Newsgroups data set is a collection of a number of newspapers group documents,
# partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge,
# it was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews paper, 
# though he does not explicitly mention this collection. 
# The 20 newsgroups collection has become a popular data set for experiments in text applications 
# of machine learning techniques, such as text classification and text clustering.

# ###### Computer
# - Graphics
# - os.ms-windows.misc
# - sys.ibm.pc.hardware
# - sys.mac.hardware
# - windows.x
# ##### Record
# - Autos
# - Motorcycles
# - sport.baseball
# - sport.hockey
# ###### Scientific
# - crypt
# - electronics
# - media
# - space
# ###### talk
# - politics.guns
# - politics.mideast
# - politics.misc
# - religion.misc
# ###### Society
# - religion. Christian
# ###### Other
# - forsale
# ###### alt
# - atheism

# Environment Setup & Dataset Loading:

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')


# Preparing & Preprocessing the Data:


vectorizer = TfidfVectorizer() #Convert a collection of raw documents to a matrix of TF-IDF features. 
# Inverse Document Frequency (TFIDF) is a technique for text vectorization based on the Bag of words (BoW) model. 
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
print(vectors_train.shape)


# Visualizing the Data


print('Training data size:', len(newsgroups_train['data']))
print('Test data size:', len(newsgroups_test['data']))


#to return the number of appearance (frequency) of each news group
targets, frequency = np.unique(newsgroups_train.target, return_counts=True)
targets_str = np.array(newsgroups_train.target_names)
print(list(zip(targets_str, frequency)))


fig=plt.figure(figsize=(10, 5), dpi= 100)
plt.bar(targets_str,frequency)
plt.xticks(rotation=90)
plt.title('Class distribution of 20 Newsgroups Training Data')
plt.xlabel('News Group')
plt.ylabel('Frequency')
plt.show()

pd.DataFrame({'data': newsgroups_train.data, 'target': newsgroups_train.target})


targets_test, frequency_test = np.unique(newsgroups_test.target, return_counts=True)
targets_test_str = np.array(newsgroups_test.target_names)
print(list(zip(targets_test_str, frequency_test)))


fig=plt.figure(figsize=(10, 5), dpi= 100)
plt.bar(targets_test_str,frequency_test)
plt.xticks(rotation=90)
plt.title('Class distribution of 20 Newsgroups Test Data')
plt.xlabel('News Group')
plt.ylabel('Frequency')
plt.show()


pd.DataFrame({'data': newsgroups_test.data, 'target': newsgroups_test.target})


# Building & Training Model:


#train the model
clf = MultinomialNB(alpha=0.01)
clf.fit(vectors_train, newsgroups_train.target)


# Evaluating Model:


# predict the group on test set
pred = clf.predict(vectors_test)
pd.DataFrame({"real value": newsgroups_test.target, "Predicted value": pred})


#accuracy 
print(accuracy_score(newsgroups_test.target, pred))

print(classification_report(newsgroups_test.target,pred))
