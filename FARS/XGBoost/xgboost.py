import pandas as pd
import numpy as np

import xgboost as xgb

import matplotlib.pylab as plt
import sys

from sklearn.metrics import roc_auc_score as auc

import random
#random.seed(0)

newseed = 1

NAVALUE = 1953

# We will drop these columns from the tables
cols_to_drop_for_accident = ['CITY', 'TWAY_ID', 'RAIL', 'MILEPT', 'MINUTE']
cols_to_drop_for_vehicle  = ['MAK_MOD', 'VIN', 'VIN_1', 'VIN_2', 'VIN_3', 'VIN_4', 'VIN_5', 'VIN_6', 'VIN_7', 'VIN_8', 'VIN_9', 'VIN_10', 'VIN_11', 'VIN_12', 'MCARR_ID', 'DR_ZIP', 'VE_FORMS']
cols_to_drop_for_person   = ['MAK_MOD', 'DEATH_TM', 'LAG_HRS', 'CERT_NO', 'MINUTE']

# Read data
acc_df = pd.read_csv('../../data/fars_train/accident_train.csv')
vehicle_df = pd.read_csv('../../data/fars_train/vehicle_train.csv')
person_df = pd.read_csv('../../data/fars_train/person_train.csv')

test_acc_df = pd.read_csv('../../data/fars_test/accident_test.csv')
test_vehicle_df = pd.read_csv('../../data/fars_test/vehicle_test.csv')
test_person_df = pd.read_csv('../../data/fars_test/person_test.csv')

# Drop some columns, fill in NaValue, and convert all to integers
acc_df = acc_df.drop(cols_to_drop_for_accident + ['YEAR'], axis=1).fillna(NAVALUE).astype(int)
vehicle_df = vehicle_df.drop(cols_to_drop_for_vehicle, axis=1).fillna(NAVALUE).astype(int)
person_df = person_df.drop(cols_to_drop_for_person, axis=1).fillna(NAVALUE).astype(int)

test_acc_df = test_acc_df.drop(cols_to_drop_for_accident, axis=1).fillna(NAVALUE).astype(int)
test_vehicle_df = test_vehicle_df.drop(cols_to_drop_for_vehicle, axis=1).fillna(NAVALUE).astype(int)
test_person_df = test_person_df.drop(cols_to_drop_for_person, axis=1).fillna(NAVALUE).astype(int)

# How many positives and negatives in accident table?
n_positive = acc_df['DRUNK_DR'].sum()
n_negative = acc_df.shape[0] - n_positive

# Join the tables
train_df = pd.merge(acc_df, vehicle_df, how='left')
train_df = pd.merge(train_df, person_df, how='left')

test_df = pd.merge(test_acc_df, test_vehicle_df, how='left')
test_df = pd.merge(test_df, test_person_df, how='left')

train_df.fillna(NAVALUE, inplace=True)
test_df.fillna(NAVALUE, inplace=True)

# Use 1/10 of training data for validation
validation_rows_acc_table = random.sample(range(acc_df.shape[0]), acc_df.shape[0] / 10)
validation_rows_acc_table.sort()
validation_rows_merged_table = train_df.ix[train_df['ID'].isin(validation_rows_acc_table)].index.tolist()
validation_id_merged_table = train_df.ix[validation_rows_merged_table]['ID']

Ytrain = train_df['DRUNK_DR']
test_id_merged_table = test_df['ID']

Ytrain = train_df['DRUNK_DR']
Xtrain = train_df.drop(['ID', 'DRUNK_DR'], axis=1)

Xtest = test_df.drop(['ID'], axis=1)
test_id_merged_table = test_df['ID']

Xvalidate = Xtrain.ix[validation_rows_merged_table]
Xtrain = Xtrain.drop(validation_rows_merged_table)

Yvalidate_reduced = acc_df.ix[validation_rows_acc_table]['DRUNK_DR']
Yvalidate = Ytrain.ix[validation_rows_merged_table]
Ytrain = Ytrain.drop(validation_rows_merged_table)

print "Starts Training ..."

gbm = xgb.XGBRegressor(max_depth=8, n_estimators=500, scale_pos_weight=n_negative / n_positive, objective='binary:logistic', seed=newseed).fit(Xtrain, Ytrain, early_stopping_rounds=10, eval_metric="auc", verbose=2, eval_set=[(Xvalidate, Yvalidate)])

print "Starts Validating ..."

Yvalidate_predicted = gbm.predict(Xvalidate)
predict_df = pd.DataFrame(data={'ID': validation_id_merged_table, 'DRUNK_DR': np.ravel(Yvalidate_predicted)})
Yvalidate_predicted = predict_df.groupby('ID').mean()
validation_accuracy = auc(Yvalidate_reduced, Yvalidate_predicted)

print "Validation Accuracy is ", validation_accuracy

print "Starts Testing"

Ytest_predicted = gbm.predict(Xtest)
predict_df = pd.DataFrame(data={'ID': test_id_merged_table, 'DRUNK_DR': np.ravel(Ytest_predicted)})
submit = predict_df.groupby('ID').mean().reset_index()
submit['ID'] = submit['ID'].astype(int)
submit = submit[['ID', 'DRUNK_DR']]
submit_name = 'submit_xgboost.csv'
submit.to_csv(submit_name, index = False)

feature_importance = pd.Series(gbm.booster().get_fscore()).sort_values(ascending=False)
print feature_importance
feature_importance.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

feature_importance.to_csv("feature_importance.csv", index = True)
