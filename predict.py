import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import logging

import utils.preprocess as preprocess
import config.constants as constants

DEBUG=True

def get_reccomended_ids(df,FILE_INDEX=constants.FILE_INDEX,header=None,getList=True):
	selected_columns = preprocess.get_features()
	#Preprocess the item to be predicted
	df = preprocess.process_data(df[selected_columns[1:]])
	#Load scale model from disk
	std_scale = joblib.load(constants.FILE_SCALE_MODEL)
	#Normalize data
	test = std_scale.transform(df)
	#Load cluster model from disk
	model = joblib.load(constants.FILE_CLUSTER_MODEL)
	#Get labels of cluster model
	labels = model.labels_
	#Predict cluster no. of test data
	cluster = model.predict(test)
	df_in = pd.read_csv(FILE_INDEX,header=header)
	similar = []
	for i in xrange(len(labels)):
		if labels[i] == cluster:
			similar.append(i)

	if getList:
		return df_in.ix[similar].values[:,0].tolist()
	else:
		return df_in.ix[similar]

if (DEBUG and __name__ == '__main__'):
	print get_reccomended_ids(pd.read_csv('./data/input_data2.csv'))

