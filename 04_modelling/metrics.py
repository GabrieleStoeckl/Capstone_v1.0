import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def cross_val_scores(model, X_train, y_train, cv):
    #crossval metrics for model
    scores = ["f1", "precision", "recall", "accuracy"]
    for score in scores:
        list_scores =cross_val_score(model, X_train, y_train, cv=cv, scoring = score)
        print(list_scores.mean())
        
def test_scores(model, X_test, y_test):
    #test scores metrics for model
    y_pred = model.predict(X_test)
    scores = [f1_score, precision_score, recall_score, accuracy_score]
    for score in scores:
        print(score(y_test, y_pred))

def train_scores(model, X_train, y_train):
    #train score metrics for model
    y_pred_train = model.predict(X_train)
    scores = [f1_score, precision_score, recall_score, accuracy_score]
    for score in scores:
        print(score(y_train, y_pred_train))

