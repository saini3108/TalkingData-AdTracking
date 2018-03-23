mport os
import gc
import time

import numpy as np
import pandas as pd 
import scikitplot.plotters as skplt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

print(os.listdir("./input"))



dtype = {  'ip' : 'uint32',
           'app' : 'uint16',
           'device' : 'uint16',
           'os' : 'uint16',
           'channel' : 'uint8',
           'is_attributed' : 'uint8'}

usecol=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
cols = ['ip', 'app', 'device', 'os', 'channel']



train = pd.read_csv(
	"./input/train.csv",
	dtype=dtype,
	infer_datetime_format=True,
	usecols=usecol,
	low_memory = True,
	nrows=20000000
	)
test = pd.read_csv("./input/test.csv")

print(train.head())
print(test.head())

def mean_test_encoding(train, test, cols, target):

	for col in cols:
		test[col + '_mean_encoded'] = np.nan

	for col in cols:
		train_mean = train.groupby(col)[target].mean()
		mean = test[col].map(train_mean)
		test[col + '_mean_encoded'] = mean

	prior = train[target].mean()

	for col in cols:
		test[col + '_mean_encoded'].fillna(prior, inplace = True)

	return test

def mean_train_encoding(df, cols, target):
	y_train = df[target].values
	skf = StratifiedKFold(
		5,
		shuffle = True,
		random_state=123
		)
	for col in cols:
		df[col + '_mean_encoded'] = np.nan

	for trn_ind, val_ind in skf.split(df, y_train):
		x_train, x_val = df.iloc[trn_ind], df.iloc[val_ind]

		for col in cols:
			train_mean = x_train.groupby(col)[target].mean()
			mean = x_val[col].map(train_mean)
			df[col + '_mean_encoded'].iloc[val_ind] = mean

	prior = df[target].mean()

	for col in cols:
		df[col + '_mean_encoded'].fillna(prior, inplace = True)

	return df


y = train['is_attributed']
cols = ['app', 'channel']
target = 'is_attributed'
res_train = mean_train_encoding(train, cols, target)
res_test = mean_test_encoding(train, test, cols, target)

res_train.drop(['click_time','is_attributed'], axis = 1, inplace = True)   
res_test.drop(['click_time','click_id'], axis = 1, inplace = True)

def print_score(m, df, y):
	print('Accuracy: [Train , Val]')
	res = m.score(df, y)
	print(res)
	print('Train Confusion Matrix')
	df_train_proba = m.predict_proba(df)
	df_train_pred_indices = np.argmax(df_train_proba, axis=1)
	classes_train = np.unique(y)
	preds_train = classes_train[df_train_pred_indices]
	skplt.plot_confusion_matrix(y, preds_train)

print(res_train.head())

test_submission = pd.read_csv("./input/sample_submission.csv")
test_submission.head()

clf = RandomForestClassifier(
	n_estimators=12,
	max_depth=6,
	min_samples_leaf=100,
	max_features=0.5,
	bootstrap=False,
	n_jobs=-1,
	random_state=123
)

clf.fit(train, y)
print_score(clf, train, y)