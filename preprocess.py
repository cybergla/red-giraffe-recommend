import pandas as pd
import re
import constants

def get_features():
	df_features = pd.read_csv(constants.FILE_FEATURES,sep=',',header=None)	
	return df_features[0].values

def process_data(input_file_name):

	selected_columns = get_features()	#Only select these features

	df = pd.read_csv(input_file_name,sep=',',header=0, usecols=selected_columns)
	#Remove rows with NaN, null values
	df = df.dropna()

	#Remove rows with invalid values
	df = df[df['_source.location.lat']!=0]

	#Preprocess data
	for index, row in df.iterrows():
		
		#Replace bhk, rk etc. in type of accomodation
		df.loc[index,constants.FIELD_TYPE_OF_ACCOMODATION] = re.sub(r"\D*","",df.loc[index,constants.FIELD_TYPE_OF_ACCOMODATION])
		#Replace floor desc with number
		df.loc[index,constants.FIELD_SOURCE_FLOOR] = re.sub(r"Duplex|Ground_\d|Ground",'0',df.loc[index,constants.FIELD_SOURCE_FLOOR])
		df.loc[index,constants.FIELD_SOURCE_FLOOR] = re.sub(r"\D*","",df.loc[index,constants.FIELD_SOURCE_FLOOR])

	return df