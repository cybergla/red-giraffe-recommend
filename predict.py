import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

import utils.preprocess as preprocess
import config.constants as constants

def get_reccomended_ids(df,FILE_INDEX=constants.FILE_INDEX):
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
	df_in = pd.read_csv(FILE_INDEX)
	similar = []
	for i in xrange(len(labels)):
		if labels[i] == cluster:
			similar.append(i)
	return df_in.ix[similar].values[:,1].tolist()