
#
# MDST Ratings Analysis Challenge
# LSTM
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

import cPickle
import gzip

import os

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from theano.tensor.nnet import sigmoid

import sklearn.linear_model
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn import svm
import sklearn.ensemble
import re

import random

np.random.seed(0)

random.seed(0)

# Load in the data - pandas DataFrame objects

train = pd.read_csv('../data/train.csv')[0:117810]
test = pd.read_csv('../data/test.csv')

train['grade'] = train['grade'].fillna('N/A').str.replace('Not sure yet', 'N/A')
train['dept'] = train['dept'].fillna('N/A')
train['interest'] = train['interest'].fillna('N/A')
train['profgender'] = train['profgender'].apply(str).replace('0', 'male').replace('1', 'female')
train['profhotness'] = train['profhotness'].apply(str).replace('0', '').replace('1', 'The professor is hot')
train = ('From department ' + train['dept'] + ' I feel ' + train['interest'] + ' The Professor is ' + train['profgender'] + ' ' + train['profhotness'] + ' ' + train['comments'] +' I got grade ' + train['grade'] + 'grade ' + train['tags']).str.replace(r'[^a-zA-Z0-9 ]', '')

test['grade'] = test['grade'].fillna('N/A').replace('Not sure yet', 'N/A')
test['dept'] = test['dept'].fillna('N/A')
test['interest'] = test['interest'].fillna('N/A')
test['profgender'] = test['profgender'].apply(str).replace('0', 'male').replace('1', 'female')
test['profhotness'] = test['profhotness'].apply(str).replace('0', '').replace('1', 'The professor is hot')
test = ('From department ' + test['dept'] + ' I feel ' + test['interest'] + ' The Professor is ' + test['profgender'] + ' ' + test['profhotness'] + ' ' + test['comments'] +' I got grade ' + test['grade'] + 'grade ' + test['tags']).str.replace(r'[^a-zA-Z0-9 ]', '')

y_train = pd.read_csv('../data/train.csv')[0:117810]['quality']

print train

count_vect = CountVectorizer(min_df=0,stop_words=None,ngram_range=(1,1), token_pattern=r"\b\w+\b")
x_comments_train = count_vect.fit_transform(train.fillna('')).toarray()

comments_features = count_vect.get_feature_names()
comments_features.append('nan')

a = str(comments_features[10])

comments_features = [str(comments_feature) for comments_feature in comments_features]
print comments_features
print 'Index of nan is ', comments_features.index('nan')

all_train_comments = []

index = 0
for i in train.as_matrix():
    print index
    comment = []
    if len(str(i).lower().split()) == 0:
        comment.append(comments_features.index('nan'))
    for j in str(i).lower().split():
        comment.append(comments_features.index(j))
    all_train_comments.append(comment)
    index = index + 1

#exit()

all_test_comments = []

index = 0
for i in test.as_matrix():
    print index
    comment = []
    if len(str(i).lower().split()) == 0:
        comment.append(comments_features.index('nan'))
    for j in str(i).lower().split():
        if j in comments_features:
            comment.append(comments_features.index(j))
        else:
            continue
    all_test_comments.append(comment)
    index = index + 1

print all_train_comments[0:10]
print all_test_comments[0:10]

x_train = pd.Series(np.asarray(all_train_comments))
x_test = pd.Series(np.asarray(all_test_comments))

y_test = pd.Series(np.zeros(len(all_test_comments)))

print "Length of train: ", len(x_train.tolist()), len([ y-2 for y in y_train.tolist() ])

train = tuple([x_train.tolist(), [ y-2 for y in y_train.tolist() ] ])
test = tuple([x_test.tolist(), y_test.tolist()])

print "Saving to file"

f = gzip.open("rmp.pkl.gz", "w")
cPickle.dump((train, test), f)
f.close()
