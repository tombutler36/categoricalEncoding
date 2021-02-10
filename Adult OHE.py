import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

#Reading full data set for Adult x test
adult_test_data = 'x_test.csv'
df = pd.read_csv(adult_test_data)

#reading full data for Ault x train
adult_train_data = 'x_train.csv'
dft = pd.read_csv(adult_train_data)

#Finding statistical features of numerical features as well as value counts of each categorical for x test
df_stat = df.describe()
workclass_frequency_test = df["workclass"].value_counts()
education_frequency_test = df["education"].value_counts()
marital_status_frequency_test = df["marital-status"].value_counts()
occupation_frequency_test = df["occupation"].value_counts()
relationship_frequency_test = df["relationship"].value_counts()
race_frequency_test = df["race"].value_counts()
sex_frequency_test = df["sex"].value_counts()
native_country_frequency_test = df["native-country"].value_counts()

#Finding statistical feateures of numerical features and value conts of each categorical for x train
dft_stat = dft.describe()
workclass_frequency_train = dft["workclass"].value_counts()
education_frequency_train = dft["education"].value_counts()
marital_status_frequency_train = dft["marital-status"].value_counts()
occupation_frequency_train = dft["occupation"].value_counts()
relationship_frequency_train = dft["relationship"].value_counts()
race_frequency_train = dft["race"].value_counts()
sex_frequency_train = dft["sex"].value_counts()
native_country_frequency_train = dft["native-country"].value_counts()

#One hot encoding for each categorical varibales for the x test data
ohc_workclass_test = pd.get_dummies(df["workclass"], prefix="workclass")
ohc_education_test = pd.get_dummies(df["education"], prefix="education")
ohc_marital_status_test = pd.get_dummies(df["marital-status"], prefix="marital_status")
ohc_occupation_test = pd.get_dummies(df["occupation"], prefix="occupation")
ohc_relationship_test = pd.get_dummies(df["relationship"], prefix="relationship")
ohc_race_test = pd.get_dummies(df["race"], prefix="race")
ohc_sex_test = pd.get_dummies(df["sex"], prefix="sex")
ohc_native_country_test = pd.get_dummies(df["native-country"], prefix="native_country")

#One hot encoding each categroical variable for the x train data
ohc_workclass_train = pd.get_dummies(dft["workclass"], prefix="workclass")
ohc_education_train = pd.get_dummies(dft["education"], prefix="education")
ohc_marital_status_train = pd.get_dummies(dft["marital-status"], prefix="marital_status")
ohc_occupation_train = pd.get_dummies(dft["occupation"], prefix="occupation")
ohc_relationship_train = pd.get_dummies(dft["relationship"], prefix="relationship")
ohc_race_train = pd.get_dummies(dft["race"], prefix="race")
ohc_sex_train = pd.get_dummies(dft["sex"], prefix="sex")
ohc_native_country_train = pd.get_dummies(dft["native-country"], prefix="native_country")


#Representing only numerical values for x test
df_numeric = df.select_dtypes(include=np.number)

#preresenting only numerical values for x train
dft_numeric = dft.select_dtypes(include=np.number)

#Standardizing and normalizing numeric data for x test
#normalize
x = df_numeric
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
numeric_scaled_test = pd.DataFrame(x_scaled)
numeric_normalized_test = numeric_scaled_test.rename(columns = {0:"age_normal",1:"fnlwgt_normal",2:"education_num_normal",3:"capital_gain_normal",4:"capital_loss_normal",5:"hours_per_week_normal"})
#standardize
scaler_test = preprocessing.StandardScaler()
numeric_standardized_test = scaler_test.fit_transform(df_numeric)
numeric_standardized_test = pd.DataFrame(numeric_standardized_test)
numeric_standardized_test = numeric_standardized_test.rename(columns = {0:"age_normal",1:"fnlwgt_normal",2:"education_num_normal",3:"capital_gain_normal",4:"capital_loss_normal",5:"hours_per_week_normal"})

#Standardizing and normalizing numeric data for x trainn
#normalize
x2 = dft_numeric
min_max_scaler_2 = preprocessing.MinMaxScaler()
x2_scaled = min_max_scaler_2.fit_transform(x2)
numeric_scaled_train = pd.DataFrame(x2_scaled)
numeric_normalized_train = numeric_scaled_train.rename(columns = {0:"age_normal",1:"fnlwgt_normal",2:"education_num_normal",3:"capital_gain_normal",4:"capital_loss_normal",5:"hours_per_week_normal"})
#standardize
scaler_train = preprocessing.StandardScaler()
numeric_standardized_train = scaler_train.fit_transform(dft_numeric)
numeric_standardized_train = pd.DataFrame(numeric_standardized_train)
numeric_standardized_train = numeric_standardized_train.rename(columns = {0:"age_normal",1:"fnlwgt_normal",2:"education_num_normal",3:"capital_gain_normal",4:"capital_loss_normal",5:"hours_per_week_normal"})


#Exporting One hot encoded data for x test (copy here)

#creating full x train encoded data set
X_test = pd.concat((ohc_education_test,ohc_marital_status_test,ohc_native_country_test,ohc_occupation_test,ohc_race_test,ohc_relationship_test,ohc_sex_test,ohc_workclass_test,numeric_normalized_test),axis=1,ignore_index=False)
X_train = pd.concat((ohc_education_train,ohc_marital_status_train,ohc_native_country_train,ohc_occupation_train,ohc_race_train,ohc_relationship_train,ohc_sex_train,ohc_workclass_train,numeric_normalized_train),axis=1,ignore_index=False)
X_test.insert(38,"native_country_Holand-Netherlands",0)

#creating decision tree classifier model
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
dt = DecisionTreeClassifier(min_samples_split=1000)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Decision Tree:')
print("Accuracy:", metrics.accuracy_score(y_test, y_pred),
      "Precision:", metrics.precision_score(y_test, y_pred),
      "Recall:", metrics.recall_score(y_test, y_pred),
      "F-Beta:", metrics.fbeta_score(y_test, y_pred,1))

rf = RandomForestClassifier()
rf.fit(X_train, y_train.values.ravel())
y_pred2 = rf.predict(X_test)
print('Random Forest:')
print("Accuracy:", metrics.accuracy_score(y_test, y_pred2),
      "Precision:", metrics.precision_score(y_test, y_pred2),
      "Recall:", metrics.recall_score(y_test, y_pred2),
      "F-Beta:", metrics.fbeta_score(y_test, y_pred2,1))
