#Need this is resolve unicode encoding errors in parsing JSON to CSV
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import json
from pandas.io.json import json_normalize

def convert_json_to_csv(input_file='file1.json',output_file='../data/raw.csv'):
	with open(input_file) as data_file:    
	    data = json.load(data_file)

	result = pd.DataFrame()
	for x in xrange(len(data['hits']['hits'])):
		result = result.append(json_normalize(data['hits']['hits'][x]))
	result.to_csv(output_file)
	return

if __name__ == "__main__":
	if (len(argv)<3):
		print "Usage: json_convert.py <input_file_name> <output_file_name>"
	else:
    	convert_json_to_csv(sys.argv[1],sys.argv[2])