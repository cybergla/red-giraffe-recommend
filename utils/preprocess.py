import pandas as pd
import re
import sys
sys.path.append('..')
import config.constants as constants

DEBUG=False

def get_features(input_file_name=constants.FILE_FEATURES):
	"""
	Get features as a list from csv
	"""
	df_features = pd.read_csv(input_file_name,sep=',',header=None)	
	return df_features[0].values

def process_data(df):
	""" 
	Sanatizes and prepares input 
	"""

	#Remove rows with NaN, null values
	df = df.dropna()

	#Remove rows with invalid values
	df = df[df[constants.FIELD_LATITUDE]!=0]

	df_features = pd.read_csv(constants.FILE_FEATURES,sep=',',header=None)
	repl_cols = []
	for index, row in df_features.iterrows():
		if (str(row[1])!='nan'):
			column = row[0]
			lst = row.values.tolist()
			lst = [x for x in lst if str(x)!='nan']
			lst = ["" if x == 'null' else x for x in lst]
			repl_cols.append(lst)

	#Preprocess data
	for index, row in df.iterrows():
		for lst in repl_cols:
			for i in xrange(1,len(lst),2):
				pattern = re.compile(lst[i])
				repl = lst[i+1]
				df.loc[index,lst[0]] = re.sub(pattern,repl,df.loc[index,lst[0]])

	#Remake index to account for dropped rows
	df.index = range(df.shape[0])

	return df


def get_data(input_file_name=constants.FILE_DATA,column_offset=0,to_index=True):

	selected_columns = get_features()	#Only select these features

	df = pd.read_csv(input_file_name,sep=',',header=0, usecols=selected_columns)
	
	if(DEBUG):
		df.to_csv('raw.csv')

	df = process_data(df)

	if(DEBUG):
		df.to_csv('preprocessed.csv')

	if(to_index):
		outf = df[selected_columns[0]]		#select only ids
		outf.to_csv(constants.FILE_INDEX,index=False)

	return df[selected_columns[column_offset:]]

