import pandas as pd
import re
import kmeans.config.constants as constants

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

	#Preprocess data
	for index, row in df.iterrows():
		
		#Replace bhk, rk etc. in type of accomodation
		df.loc[index,constants.FIELD_TYPE_OF_ACCOMODATION] = re.sub(r"\D","",df.loc[index,constants.FIELD_TYPE_OF_ACCOMODATION])
		#Replace floor desc with number
		df.loc[index,constants.FIELD_SOURCE_FLOOR] = re.sub(r"Duplex|Ground_\d|Ground",'0',df.loc[index,constants.FIELD_SOURCE_FLOOR])
		df.loc[index,constants.FIELD_SOURCE_FLOOR] = re.sub(r"\D","",df.loc[index,constants.FIELD_SOURCE_FLOOR])

	#Remake index to account for dropped rows
	df.index = range(df.shape[0])

	if(DEBUG):
		df.to_csv('preprocessed.csv')

	return df


def get_data(input_file_name=constants.FILE_DATA,column_offset=0,to_index=True):

	selected_columns = get_features()	#Only select these features

	df = pd.read_csv(input_file_name,sep=',',header=0, usecols=selected_columns)
	
	if(DEBUG):
		df.to_csv('raw.csv')

	df = process_data(df)

	if(to_index):
		outf = df[selected_columns[0]]		#select only ids
		outf.to_csv(constants.FILE_INDEX)

	return df[selected_columns[column_offset:]]