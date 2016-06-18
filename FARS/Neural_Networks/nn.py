import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

import network4
from network4 import Network
from network4 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, LinearLayer

import random
#random.seed(0)

mini_batch_size = 83
learning_rate = 0.01
lmbda = 0.1

cols_to_keep_for_accident = ['ID', 'STATE', 'VE_FORMS', 'PEDS', 'PERSONS', 'COUNTY', 'DAY_WEEK', 'HOUR', 'ROAD_FNC', 'HARM_EV', 'MAN_COLL', 'REL_ROAD', 'LGT_COND', 'NOT_HOUR', 'ARR_HOUR', 'HOSP_HR', 'FATALS', 'CF1', 'CF2', 'CF3', 'MONTH', 'NHS', 'SP_JUR', 'WEATHER', 'SCH_BUS', 'LATITUDE', 'LONGITUD', 'DRUNK_DR']
#cols_to_keep_for_vehicle = ['ID', 'MODEL', 'MAKE', 'BODY_TYP', 'MOD_YEAR', 'M_HARM', 'L_COMPL', 'PREV_SUS', 'PREV_DWI', 'DEATHS']
cols_to_keep_for_vehicle = ['ID', 'MAKE', 'BODY_TYP', 'MOD_YEAR', 'M_HARM', 'L_COMPL', 'L_STATUS', 'PREV_SUS', 'PREV_DWI', 'REG_STAT', 'L_RESTRI', 'CDL_STAT', 'LAST_MO', 'LAST_YR', 'FIRST_MO', 'FIRST_YR', 'HIT_RUN', 'OWNER', 'TOW_VEH', 'J_KNIFE', 'GVWR', 'V_CONFIG', 'CARGO_BT', 'BUS_USE', 'SPEC_USE', 'EMER_USE', 'TRAV_SP', 'UNDERIDE', 'IMPACT1', 'DEFORMED', 'FIRE_EXP', 'DR_PRES', 'L_ENDORS', 'DR_HGT', 'DR_WGT', 'PREV_ACC', 'PREV_SPD', 'PREV_OTH']
#cols_to_keep_for_person = ['ID', 'VEH_NO', 'PER_NO', 'MAKE', 'BODY_TYP', 'MOD_YEAR', 'AGE', 'SEX', 'EJECTION', 'EJ_PATH', 'DEATH_HR']
cols_to_keep_for_person = ['ID', 'VEH_NO', 'MAKE', 'BODY_TYP', 'MOD_YEAR', 'AGE', 'SEX', 'DEATH_HR', 'ROLLOVER', 'INJ_SEV', 'REST_USE', 'EJECTION', 'EJ_PATH', 'DOA', 'DEATH_DA', 'LOCATION', 'RACE', 'WORK_INJ', 'PER_NO', 'PER_TYP', 'SEAT_POS', 'AIR_BAG', 'EXTRICAT', 'HOSPITAL', 'DEATH_MO', 'HISPANIC']

acc_df = pd.read_csv('../../data/fars_train/accident_train.csv')[cols_to_keep_for_accident].fillna(0).astype(int)
vehicle_df = pd.read_csv('../../data/fars_train/vehicle_train.csv')[cols_to_keep_for_vehicle].fillna(0).astype(int)
person_df = pd.read_csv('../../data/fars_train/person_train.csv')[cols_to_keep_for_person].fillna(0).astype(int)

del cols_to_keep_for_accident[-1] # No 'DRUNK_DR' in test data
test_acc_df = pd.read_csv('../../data/fars_test/accident_test.csv')[cols_to_keep_for_accident].fillna(0).astype(int)
test_vehicle_df = pd.read_csv('../../data/fars_test/vehicle_test.csv')[cols_to_keep_for_vehicle].fillna(0).astype(int)
test_person_df = pd.read_csv('../../data/fars_test/person_test.csv')[cols_to_keep_for_person].fillna(0).astype(int)

train_df = pd.merge(acc_df, vehicle_df, on='ID', how='left')
train_df = pd.merge(train_df, person_df, on=['ID', 'MAKE', 'BODY_TYP', 'MOD_YEAR'], how='left')

test_df = pd.merge(test_acc_df, test_vehicle_df, on='ID', how='left')
test_df = pd.merge(test_df, test_person_df, on=['ID', 'MAKE', 'BODY_TYP', 'MOD_YEAR'], how='left')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

originial_validation_rows = random.sample(range(acc_df.shape[0]), acc_df.shape[0] / 10)
validation_rows = train_df[train_df['ID'].isin(originial_validation_rows)].index.tolist()
validate_id = train_df.ix[validation_rows]['ID']

Ydata = train_df['DRUNK_DR'] # Does have duplicated info about 'DRUNK_DR'
test_id = test_df['ID']


#cat_cols = ['STATE', 'VE_FORMS', 'PEDS', 'PERSONS', 'DAY_WEEK', 'HOUR', 'ROAD_FNC', 'HARM_EV', 'MAN_COLL', 'REL_ROAD', 'LGT_COND', 'NOT_HOUR', 'ARR_HOUR', 'HOSP_HR', 'FATALS']
cat_cols = list(test_df.columns.values)
cat_cols.remove('ID')
print cat_cols

df = pd.concat([train_df, test_df], axis= 0)

print df.shape[0], df.shape[1]

df_category = pd.get_dummies(df[cat_cols], columns = cat_cols, prefix = cat_cols, prefix_sep = ".")

Xdata = df_category[:train_df.shape[0]]
Xtest = df_category[train_df.shape[0]:]

#validation_rows = random.sample(Xdata.index, 65658)
Xvalidate = Xdata.ix[validation_rows]
Xtrain = Xdata.drop(validation_rows)

Yvalidation = Ydata.ix[validation_rows]
Ytrain = Ydata.drop(validation_rows)

Xvalidate = Xvalidate[:len(validation_rows) / mini_batch_size * mini_batch_size]
Yvalidation = Yvalidation[:len(validation_rows) / mini_batch_size * mini_batch_size]
validate_id = validate_id[:len(validation_rows) / mini_batch_size * mini_batch_size]

afn = sigmoid

net = Network([
        FullyConnectedLayer(n_in=Xtrain.shape[1], n_out=50, activation_fn=afn),
        SoftmaxLayer(n_in=50, n_out=2)], mini_batch_size)
#        SoftmaxLayer(n_in=Xtrain.shape[1],n_out=2)], mini_batch_size)

net.SGD(Xtrain, Ytrain, 200, mini_batch_size, learning_rate,
        Xvalidate, Yvalidation, Xtest, test_id, validate_id, lmbda)
