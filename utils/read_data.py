import pandas as pd
import sys
import json
from pandas.io.json import json_normalize
sys.path.append('.')
sys.path.append('..')

def read(input_file_name,source_type,selected_columns):
	#Get data from CSV file
	if(source_type=='csv'):
		df = pd.read_csv(input_file_name,sep=',',header=0, usecols=selected_columns)
		return df;
	#Get data from JSON file
	elif(source_type=='json'):
		#Read JSON
		with open(input_file_name) as data_file:    
			data = json.load(data_file)
		result = pd.DataFrame()
		#Normalize JSON, only select the list of property details
		for x in xrange(len(data['hits']['hits'])):
			result = result.append(json_normalize(data['hits']['hits'][x]))
		df = pd.DataFrame(result, columns=selected_columns);			
		return df;
