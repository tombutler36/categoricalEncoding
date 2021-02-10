import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


#splitting numeric data away from categorical
def split_numeric(dataframe):
    numeric = dataframe.select_dtypes(include=np.number)
    return numeric
#splitting categorical
def split_categorical(dataframe):
    categorical = dataframe.select_dtypes(exclude=np.number)
    return categorical
#normalizing numeric data
def normalize(dataframe_numeric):
    name = list(dataframe_numeric.columns)
    mms = preprocessing.MinMaxScaler()
    x_scaled = mms.fit_transform(dataframe_numeric)
    numeric_normalized = x_scaled
    numeric_normalized = pd.DataFrame(numeric_normalized)
    numeric_normalized.columns = name
    return numeric_normalized
#standardizing numeric data
def standardize(dataframe_numeric):
    name = list(dataframe_numeric.columns)
    st = preprocessing.StandardScaler()
    numeric_standardized = st.fit_transform(dataframe_numeric)
    numeric_standardized = pd.DataFrame(numeric_standardized)
    numeric_standardized.columns = name
    return numeric_standardized

def ohe(cat_data):
    ohe = OneHotEncoder(sparse=False)
    ohenc = ohe.fit_transform(cat_data)
    name = ohe.get_feature_names(cat_data.columns)
    ohenc = pd.DataFrame(ohenc)
    ohenc.columns = name
    return ohenc

def concat_data(categorical,numeric):
    final = pd.concat((categorical,numeric),axis=1,ignore_index=False)
    return final

def clean_entries(dataframe_ohe):
    local_dataframe_ohe = dataframe_ohe.copy(deep=True)
    for col in local_dataframe_ohe.columns:
        if "?" not in str(col):
            continue
        else:
            del local_dataframe_ohe[col]
    return local_dataframe_ohe

def equalize_splits(X_train, X_test):
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test.insert(X_train.columns.get_loc(col), str(col), 0)
    for col in X_test.columns:
        if col not in X_train.columns:
            X_train.insert(X_test.columns.get_loc(col), str(col), 0)
    return X_train, X_test

def compositeprepare(xtrainstring, xteststring, ytrainstring, yteststring):
    #reading csv files to pandas dataframes
    xtrain = pd.read_csv(xtrainstring)
    xtest = pd.read_csv(xteststring)
    ytrain = pd.read_csv(ytrainstring)
    ytest = pd.read_csv(yteststring)
    #splitting x data to numeric and normalized values
    xtrainnumeric = split_numeric(xtrain)
    xtestnumeric = split_numeric(xtest)
    xtrainnormalized = normalize(xtrainnumeric)
    xtestnormalized = normalize(xtestnumeric)
    #splitting category data from x files
    xtraincat = split_categorical(xtrain)
    xtestcat = split_categorical(xtest)
    #ohe categorical data
    xtrainohe = ohe(xtraincat)
    xtestohe = ohe(xtestcat)
    xtrainfullenc = concat_data(xtrainohe, xtrainnormalized)
    xtestfullenc = concat_data(xtestohe, xtestnormalized)
    xtrainclean = clean_entries(xtrainfullenc)
    xtestclean = clean_entries(xtestfullenc)
    xtrain, xtest = equalize_splits(xtrainclean, xtestclean)
    return xtrain, xtest, ytrain, ytest


xtrain, xtest, ytrain, ytest = compositeprepare("x_train_amz.csv", "x_test_amz.csv", "y_train_amz.csv", "y_test_amz.csv")


dt = DecisionTreeClassifier(min_samples_split=200, criterion='entropy', max_depth=24)
dt.fit(xtrain, ytrain)
ypreddt = dt.predict(xtest)
ypreddt2 = dt.predict(xtrain)
print('Decision Tree Test:')
print("Accuracy:", metrics.accuracy_score(ytest, ypreddt),
      "Precision:", metrics.precision_score(ytest, ypreddt),
      "Recall:", metrics.recall_score(ytest, ypreddt),
      "F-Beta:", metrics.fbeta_score(ytest, ypreddt,1))

print('Decision Tree Train:')
print("Accuracy:", metrics.accuracy_score(ytrain, ypreddt2),
      "Precision:", metrics.precision_score(ytrain, ypreddt2),
      "Recall:", metrics.recall_score(ytrain, ypreddt2),
      "F-Beta:", metrics.fbeta_score(ytrain, ypreddt2,1))


rf = RandomForestClassifier(n_estimators=500, min_samples_split=1000)
rf.fit(xtrain, ytrain.values.ravel())
ypredrf = rf.predict(xtest)
ypredrf2 = rf.predict(xtrain)
print('Random Forest Test:')
print("Accuracy:", metrics.accuracy_score(ytest, ypredrf),
      "Precision:", metrics.precision_score(ytest, ypredrf),
      "Recall:", metrics.recall_score(ytest, ypredrf),
      "F-Beta:", metrics.fbeta_score(ytest, ypredrf,1))
print('Random Forest Train:')
print("Accuracy:", metrics.accuracy_score(ytrain, ypredrf2),
      "Precision:", metrics.precision_score(ytrain, ypredrf2),
      "Recall:", metrics.recall_score(ytrain, ypredrf2),
      "F-Beta:", metrics.fbeta_score(ytrain, ypredrf2,1))
