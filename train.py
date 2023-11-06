#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc

import pickle

# parameters selected while tuning
random_state=1
output_file = 'xgb_model.bin'
np.random.seed(1)

xgb_params = {
    'eta': 0.3, 
    'max_depth': 1,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

# Data Prep
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
df = pd.concat([train_csv, test_csv])
# delete the id columns, they are not useful to us
del df['Unnamed: 0']
del df['id']
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.fillna(0)
# get categorical columns and covert to list
categoricals = df.select_dtypes('object').columns.tolist()
# get numerical columns and convert to a list
numericals = df.select_dtypes('int64', 'float64').columns.tolist()
# check for the unique vales in each column
# Change the values to lower case 
df_copy = df.copy()

for category in categoricals:
    df_copy[category] = df_copy[category].str.lower().str.replace(' ', '_')

# Lets review the avg. ratings for each column
# Remove the non-rating values
non_rating_cols = ['age', 'flight_distance', 'departure_delay_in_minutes']
ratings = [col for col in numericals if col not in non_rating_cols]

unique_ratings = {col: df_copy[col].nunique() for col in ratings}

df_clean = df_copy.copy()
# lets remove the '0' values
for k, v in unique_ratings.items():
    df_clean = df_clean[(df_clean[k] != 0)]
{col: df_clean[col].nunique() for col in ratings}
# Now we have out DF_CLEAN.
df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=random_state)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Let convert the statuses to 1 and 0
# y_train values/Target values == satisfaction
# if satisfied, 1. if unsatisfied, 0
y_train = (df_train.satisfaction == 'satisfied').astype('int').values
y_test = (df_test.satisfaction == 'satisfied').astype('int').values

del df_train['satisfaction']
del df_test['satisfaction']
del categoricals['satisfaction']

train_dicts = df_train.fillna(0).to_dict(orient='records')
test_dicts = df_test.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)
features = list(dv.get_feature_names_out())

# DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# function to train the model
def train(df_train, y_train, n_estimators, max_depth, min_samples_leaf, bootstrap, random_state):
    
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      bootstrap=bootstrap,
                                      random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    return dv, rf_model

# function to predict using the model and DictVectorizer
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


dv, rf_model = train(df_train, y_train, n_estimators, max_depth, min_samples_leaf, bootstrap, random_state)

# saving the DictVectorizer and RandomForest model to the same file
with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, rf_model), f_out)
