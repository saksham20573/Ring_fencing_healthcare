import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression





def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


def train_model(df):
    corr_features = correlation(df, 0.35)

    X_corr = df.drop(corr_features,axis=1)

    # Train-Test Split
    X = df.drop('stroke', axis=1)
    y = df['stroke'] 
        
    # X_train, X_test, y_train, y_test = train_test_split(X_corr, y, test_size=0.3, stratify=y, random_state=42)

    # Create a SMOTE object
    smote = SMOTE(random_state=42)

    # Fit the SMOTE object to the training data and oversample the minority class
    X_smote, y_smote = smote.fit_resample(X, y)

    # Logistic Regression

    clf_LR = LogisticRegression()
    clf_LR.fit(X_smote,y_smote)
    return clf_LR

def predict(model, input):
    return model.predict(input)