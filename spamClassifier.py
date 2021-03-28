# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:25:38 2021

@author: Dhanasekar
"""

import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep= '\t', names=['labels', 'messages'])

#Data cleaning and preprocessing
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

lemmatization = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['messages'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatization.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['labels'])
y = y.iloc[:, 1].values

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
con_matrix = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)




















