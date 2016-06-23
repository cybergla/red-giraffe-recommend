import pandas as pd
import sys
import json
from pandas.io.json import json_normalize
sys.path.append('.')
sys.path.append('..')

def read(input_file_name,source_type,selected_columns):
	
	if(source_type=='csv'):
		df = pd.read_csv(input_file_name,sep=',',header=0, usecols=selected_columns)
		return df;
	elif(source_type=='json'):
		with open(input_file_name) as data_file:    
			data = json.load(data_file)	
		result = pd.DataFrame()
		for x in xrange(len(data['hits']['hits'])):
			result = result.append(json_normalize(data['hits']['hits'][x]))
		df = pd.DataFrame(result, columns=selected_columns);			
		return df;
