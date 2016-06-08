#Need this is resolve unicode encoding errors in parsing JSON to CSV
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import json
from pandas.io.json import json_normalize

with open('file1.json') as data_file:    
    data = json.load(data_file)

result = pd.DataFrame()
for x in xrange(len(data['hits']['hits'])):
	result = result.append(json_normalize(data['hits']['hits'][x]))
result.to_csv("raw.csv")