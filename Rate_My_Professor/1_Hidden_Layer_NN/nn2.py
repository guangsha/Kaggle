
#
# MDST Ratings Analysis Challenge
# SVM
#
# Guangsha Shi
#
#
# Prerequisites:
#
# numpy
# pandas
# sklearn
#

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from theano.tensor.nnet import sigmoid

import network4
from network4 import Network
from network4 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, LinearLayer

import sklearn.linear_model
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
import sklearn.ensemble
import re

import random

#np.random.seed(0)

random.seed(0)

# Load in the data - pandas DataFrame objects

train = pd.read_csv('../data/train.csv')[0:117810]
test = pd.read_csv('../data/test.csv')

# double check id and tid
#train.id[train.id.isnull()] = 0
#train.tid[train.tid.isnull()] = 0

#test.id[test.id.isnull()] = 0
#test.tid[test.tid.isnull()] = 0

# Convert date to date difference
train.date = pd.to_datetime(train.date)
train_latest_date = train.date.max()
train.date = (train_latest_date - train.date).dt.days

test.date = pd.to_datetime(test.date)
test.date = (train_latest_date - test.date).dt.days

# double check helpcount, nothelpcount, profgender, profhotness
#train.helpcount[train.helpcount.isnull()] = 0
#train.nothelpcount[train.nothelpcount.isnull()] = 0
#train.profgender[train.profgender.isnull()] = 0
#train.profhotness[train.profhotness.isnull()] = 0

#test.helpcount[test.helpcount.isnull()] = 0
#test.nothelpcount[test.nothelpcount.isnull()] = 0
#test.profgender[test.profgender.isnull()] = 0
#test.profhotness[test.profhotness.isnull()] = 0

# numeric x
numeric_cols = ['id', 'tid', 'date', 'helpcount', 'nothelpcount', 'profgender', 'profhotness']
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>
max_train = np.amax( x_num_train, 0 ).astype(np.float)
x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train # scale test by max_train

#print x_num_train

# y
y_train = train.quality

# categorical
cat_train = train.drop( numeric_cols + ['quality', 'helpfulness', 'clarity', 'easiness', 'comments'], axis = 1 )
cat_test = test.drop( numeric_cols + ['comments'], axis = 1 )

cat_train = cat_train.fillna('N/A') 
cat_test = cat_test.fillna('N/A')

# Remove + and - from grades
cat_train['grade'] = cat_train['grade'].str.replace('+', '')
cat_train['grade'] = cat_train['grade'].str.replace('-', '')
cat_train['grade'] = cat_train['grade'].str.replace('Not sure yet', 'N/A')
#cat_train['grade'] = cat_train['grade'].fillna('N/A')
print cat_train['grade']

# Handle the tags
cat_train['tags'] = cat_train['tags'].str.replace('\[\"', '')
cat_train['tags'] = cat_train['tags'].str.replace('\"\]', '')

cat_test['tags']  = cat_test['tags'].str.replace('\[\"', '')
cat_test['tags']  = cat_test['tags'].str.replace('\"\]', '')

df_train_tags = cat_train['tags'].str.split('", "').apply(pd.Series)
df_train_tags.index = train.id
df_train_tags = df_train_tags.stack().reset_index(level='id').reset_index(drop=True)
df_train_tags.columns = ['id', 'ttags']

df_test_tags = cat_test['tags'].str.split('", "').apply(pd.Series)
df_test_tags.index = test.id
df_test_tags = df_test_tags.stack().reset_index(level='id').reset_index(drop=True)
df_test_tags.columns = ['id', 'ttags']

print df_train_tags.head(10)

vvectorizer = DV( sparse = False )
vector_tags_train = vvectorizer.fit_transform( df_train_tags.drop(['id'], axis=1).T.to_dict().values() )
vector_tags_test  = vvectorizer.transform( df_test_tags.drop(['id'], axis=1).T.to_dict().values() ) 

tags_features = vvectorizer.get_feature_names()
print tags_features
vector_id_train = df_train_tags.drop(['ttags'], axis=1).as_matrix()
vector_id_test  = df_test_tags.drop(['ttags'], axis=1).as_matrix()

df_train_id = pd.DataFrame(vector_id_train, columns=['id'])
df_train_tags = pd.DataFrame(vector_tags_train, columns=tags_features)
df_train_tags = df_train_id.join(df_train_tags)
df_train_tags = df_train_tags.groupby(df_train_tags.id).sum()

df_test_id = pd.DataFrame(vector_id_test, columns=['id'])
df_test_tags = pd.DataFrame(vector_tags_test, columns=tags_features)
df_test_tags = df_test_id.join(df_test_tags)
df_test_tags = df_test_tags.groupby(df_test_tags.id).sum()

print "Haha1"

df_train_tags = train['id'].to_frame(name='id').join(df_train_tags, on='id')
x_tags_train = df_train_tags.ix[:, 1:].as_matrix()

print "Haha2"

df_test_tags = test['id'].to_frame(name='id').join(df_test_tags, on='id')
x_tags_test = df_test_tags.ix[:, 1:].as_matrix()

#print df_train_tags

# Now I am done with tags, so it's time to drop them
cat_train = cat_train.drop( ['tags'], axis = 1 )
cat_test = cat_test.drop( ['tags'], axis = 1 )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize
vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# comments
count_vect = CountVectorizer(min_df=100,stop_words=None,ngram_range=(1,2))
x_comments_train = count_vect.fit_transform(train.comments.fillna('')).toarray()
x_comments_test  = count_vect.transform(test.comments.fillna('')).toarray()

# complete x
x_train = np.hstack(( x_num_train, vec_x_cat_train, x_tags_train, x_comments_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test, x_tags_test, x_comments_test ))

n_col = x_train.shape[1]
print "Number of features: ", n_col

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

validation_rows = random.sample(x_train.index, 14000)
print "Double check if random has a seed: ", validation_rows[0], validation_rows[2000], validation_rows[4000], validation_rows[6000], validation_rows[8000]
x_real_validation = x_train.ix[validation_rows]
x_real_train = x_train.drop(validation_rows)

y_train = pd.Series(y_train)

y_real_validation = y_train.ix[validation_rows]
y_real_train = y_train.drop(validation_rows)

print "Finished data processing"

print "Now starts training"

mini_batch_size = 7

afn = sigmoid

net = Network([
        FullyConnectedLayer(n_in=n_col, n_out=100, activation_fn=afn),
#        FullyConnectedLayer(n_in=100, n_out=100, activation_fn=afn),
        SoftmaxLayer(n_in=100, n_out=9)], mini_batch_size)

net.SGD(x_real_train, y_real_train, 100, mini_batch_size, 0.003,
        x_real_validation, y_real_validation, x_test, test.id, 0.1)

#submit = pd.DataFrame(data={'id': test.id, 'quality': y_test})
#submit.to_csv('submit_ridge_vectorized.csv', index = False)
