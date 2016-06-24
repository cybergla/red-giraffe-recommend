import pandas as pd
import re
import logging
import sys
sys.path.append('.')
sys.path.append('..')
import sqlite3

import read_data
import config.constants as constants

log = logging.getLogger('recommend.preprocess')

DEBUG=False

def get_features(input_file_name=constants.FILE_FEATURES):
	"""
	Get features as a list from csv
	"""
	df_features = pd.read_csv(input_file_name,sep=',',header=None)	
	return df_features[0].values

def drop_data(df,coulmn,invalid):
	#Remove rows with invalid values
	df = df[df[coulmn]!=invalid]
	return df

def process_data(df):
	""" 
	Sanatizes and prepares input 
	"""
	log.info("Preprocessing")
	log.debug("Initial: n_features: %d n_samples: %d" % (df.shape[1],df.shape[0]))

	#Remove rows with NaN, null values
	df = df.dropna()

	log.info("Reading features from %s" % constants.FILE_FEATURES)
	df_features = pd.read_csv(constants.FILE_FEATURES,sep=',',header=None)

	repl_cols = []
	for index, row in df_features.iterrows():
		if (str(row[1])!='nan'):
			df=drop_data(df,row[0],row[1])
		if (str(row[2])!='nan'):
			column = row[0]
			lst = row.values.tolist()
			lst = [x for x in lst if str(x)!='nan']
			lst = ["" if x == 'null' else x for x in lst]
			repl_cols.append(lst)

	log.debug("Number of processed features: %d" % len(repl_cols))
	#Remake index to account for dropped rows
	df.index = range(df.shape[0])
	#Preprocess data
	for index, row in df.iterrows():
		for lst in repl_cols:
			for i in xrange(1,len(lst),2):
				pattern = re.compile(lst[i])
				repl = lst[i+1]
				df.loc[index,lst[0]] = re.sub(pattern,str(repl),df.loc[index,lst[0]])

	log.debug("Final: n_features: %d n_samples: %d" % (df.shape[1],df.shape[0]))

	return df


def get_data(input_file_name=constants.FILE_DATA,column_offset=0,source_type=constants.SOURCE_TYPE,to_index=True,to_append=False):

	selected_columns = get_features()	#Only select these features

	log.info("Reading data")

	try:
		df = read_data.read(input_file_name,source_type,selected_columns)
	except IOError as e:
		log.error("Could not open file - %s" % input_file_name)
		raise
	
	if(DEBUG):
		df.to_csv('raw.csv')

	df = process_data(df)

	if(DEBUG):
		df.to_csv('preprocessed.csv')

	if(to_index):
		log.info("Connecting to db: %s" % constants.FILE_INDEX_DB)
		conn = sqlite3.connect(constants.FILE_INDEX_DB)
		outf = df[selected_columns[0]]	#select only ids
		if(to_append):
			log.info("Appending to index")
			outf.to_sql('ids',conn,if_exists='append',index_label="indx")
		else:
			log.info("Saving to index")
			outf.to_sql('ids',conn,if_exists='replace',index_label="indx")
		conn.close()
		
	return df[selected_columns[column_offset:]]

