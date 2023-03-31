# -*- coding: utf-8 -*-
"""Analysis_classifier_results.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aovrpGtfspmGA0gviE3wjf0vTpscxLXF
"""

!pip install sentence-transformers

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("/content/drive/MyDrive/Ring_fencing_files/Analysis_Classification_dataset - Sheet1.csv")

def get_bert_embedding(data_frame):
  """
  Input a data frame and return the bert embedding vectors for the each sentence column.
  Return 2 matrices each of shape (#_samples, #size_of_word_emb).
  """
  cont_model = SentenceTransformer('distilbert-base-uncased')
  
  feature1 = cont_model.encode(data_frame)
  
  return feature1

column = "LABEL"
df_enc = df.copy()
le = preprocessing.LabelEncoder()
le.fit(df[column].unique())
df_enc[column] = le.transform(df[column])

"""k-fold cross validation"""

def run_4_folds(clf, df):
  size = len(df.index) // 4
  start = 0
  folds = []
  df = df.sample(frac = 1)
  for i in range(3):
    folds.append(df.iloc[start:start + size, :])
    start += size
  folds.append(df.iloc[start:, :])
  f1_scores = []
  accuracies = []
  prec = []
  rec = []
  for i in range(4):
    temp = folds.copy()
    df_test = temp.pop(i)
    df_train = pd.concat(temp)
    X_train = df_train["QUERY"]
    y_train = df_train["LABEL"]
    X_test = df_test["QUERY"]
    y_test = df_test["LABEL"]
    feature_1_train = get_bert_embedding(np.array(X_train))
    if clf == "svc":
      model_classify = SVC()
    if clf == "lr":
      model_classify = LogisticRegression(max_iter = 500)
    if clf == "mlp":
      model_classify = MLPClassifier(hidden_layer_sizes = (256, 128, 64), activation = "logistic")
    if clf == "dt":
      model_classify = DecisionTreeClassifier(criterion = "entropy")
    model_classify.fit(np.array(feature_1_train), y_train)
    feature_1_test = get_bert_embedding(np.array(X_test))
    preds = model_classify.predict(feature_1_test)
    f1_scores.append(f1_score(y_test, preds, average = "macro"))
    accuracies.append(accuracy_score(y_test, preds))
    prec.append(precision_score(y_test, preds, average = "macro"))
    rec.append(recall_score(y_test, preds, average = "macro"))
  return sum(f1_scores) / 4, sum(accuracies) / 4, sum(prec) / 4, sum(rec) / 4

"""SVC"""

f1, acc, prec, rec = run_4_folds("svc", df_enc)

print("F1 score: " + str(f1))
print("Accuracy: " + str(acc))
print("Precision: "+ str(prec))
print("Recall: "+ str(rec))

"""Logistic regression"""

f1, acc, prec, rec = run_4_folds("lr", df_enc)

print("F1 score: " + str(f1))
print("Accuracy: " + str(acc))
print("Precision: "+ str(prec))
print("Recall: "+ str(rec))

"""MLP classifier"""

f1, acc, prec, rec = run_4_folds("mlp", df_enc)

print("F1 score: " + str(f1))
print("Accuracy: " + str(acc))
print("Precision: "+ str(prec))
print("Recall: "+ str(rec))

"""Decision Tree"""

f1, acc, prec, rec = run_4_folds("dt", df_enc)

print("F1 score: " + str(f1))
print("Accuracy: " + str(acc))
print("Precision: "+ str(prec))
print("Recall: "+ str(rec))