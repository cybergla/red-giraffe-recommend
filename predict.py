import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import logging
import sqlite3

import utils.preprocess as preprocess
import config.constants as constants
import utils.scaling as scaling
DEBUG=True

def get_reccomended_ids(df,FILE_INDEX=constants.FILE_INDEX,header=None):
	#Get required features
	selected_columns = preprocess.get_features()
	#Preprocess the item to be predicted
	df = preprocess.process_data(df[selected_columns[1:]])
	#Normalize data
	test = scaling.scale_new_data(df)
	#Load cluster model from disk
	model = joblib.load(constants.FILE_CLUSTER_MODEL)
	#Get labels of cluster model
	labels = model.labels_
	#Predict cluster no. of test data
	cluster = model.predict(test)
	#Get list of similar item labels
	similar = []
	for i in xrange(len(labels)):
		if labels[i] == cluster:
			similar.append(i)
	#Open connection to DB
	conn = sqlite3.connect(constants.FILE_INDEX_DB)
	#Get db cursor
	cur = conn.cursor()
	#Create SQL query
	sql="select _id from ids where indx in ({seq})".format(seq=','.join(['?']*len(similar)))
	#Run query
	cur.execute(sql,similar)
	#Fetch and return result
	result = cur.fetchall()
	if result == []:
		return None
	return [r[0] for r in result] 	#Convert the list of tuples into a simple list

#print get_reccomended_ids(pd.read_csv('input_data.csv'))#.to_csv('results.csv')

if (DEBUG and __name__ == '__main__'):
	print get_reccomended_ids(pd.read_csv(constants.FILE_INPUT_DATA))

