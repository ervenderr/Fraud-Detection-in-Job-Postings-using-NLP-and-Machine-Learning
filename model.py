import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix
from flask import Flask, render_template, request, jsonify, flash
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('clean_fakejobs.csv')

# Splitting dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.text, data.fraudulent, test_size=0.3)

# Converting the data into vector format
#  instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
# fit
vect.fit(X_train)

# transform training data
X_train_dtm = vect.transform(X_train)

X_test_dtm = vect.transform(X_test)

# instantiate a Decision Tree Classifier
rf = RandomForestClassifier()

clf = rf.fit(X_train_dtm, y_train)
y_pred = clf.predict(X_test_dtm)

# Saving model to disk
pickle.dump(clf, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict())
