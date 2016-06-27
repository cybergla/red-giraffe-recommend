from flask import Flask, request, jsonify
from pandas.io.json import json_normalize
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import predict
import partialfit

app = Flask(__name__)

@app.route('/test')
def return_success():
	return "Success!"

@app.route('/predict/api/v1/get-prediction/',methods=['POST'])
def return_predictions():
	if request.headers['Content-Type'] == 'application/json':
		data = request.json
		result = json_normalize(data['hits']['hits'])

		response = jsonify(predict.get_reccomended_ids(result))
		return response, '200', {"Content-Type": "application/json"}

	else:
		return "Unsupported Media Type. Expected JSON object",'415'

@app.route('/cluster/api/v1/partial-fit/',methods=['POST'])
def do_partial_fit():
	
	filename = request.form['filename']
	partialfit.fit('./data/'+filename)
	
	return "Done",'200'

if __name__ == '__main__':
	app.run()
