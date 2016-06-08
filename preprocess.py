import pandas as pd
import re
import constants

DEBUG=False

def get_features(input_file_name=constants.FILE_FEATURES):
	df_features = pd.read_csv(input_file_name,sep=',',header=None)	
	return df_features[0].values

def get_data(input_file_name=constants.FILE_DATA,column_offset=0):

	selected_columns = get_features()	#Only select these features

	df = pd.read_csv(input_file_name,sep=',',header=0, usecols=selected_columns)
	if(DEBUG):
		df.to_csv('raw.csv')

	#Remove rows with NaN, null values
	df = df.dropna()

	#Remove rows with invalid values
	df = df[df[constants.FIELD_LATITUDE]!=0]

	#Preprocess data
	for index, row in df.iterrows():
		
		#Replace bhk, rk etc. in type of accomodation
		df.loc[index,constants.FIELD_TYPE_OF_ACCOMODATION] = re.sub(r"\D*","",df.loc[index,constants.FIELD_TYPE_OF_ACCOMODATION])
		#Replace floor desc with number
		df.loc[index,constants.FIELD_SOURCE_FLOOR] = re.sub(r"Duplex|Ground_\d|Ground",'0',df.loc[index,constants.FIELD_SOURCE_FLOOR])
		df.loc[index,constants.FIELD_SOURCE_FLOOR] = re.sub(r"\D*","",df.loc[index,constants.FIELD_SOURCE_FLOOR])

	#Remake index to account for dropped rows
	df.index = range(df.shape[0])

	#outf = df[selected_columns[0]]		#select only ids
	#outf.to_csv('index.csv')

	if(DEBUG):
		df.to_csv('preprocessed.csv')
	return df[selected_columns[column_offset:]]

if(DEBUG):
	get_data('prod.csv')