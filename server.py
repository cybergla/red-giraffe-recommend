from flask import Flask, request, jsonify
from pandas.io.json import json_normalize
'''
curl -H "Content-type: application/json" -X POST http://127.0.0.1:5000/predict/api/v1/get-prediction/ -d ''
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import predict

app = Flask(__name__)

@app.route('/predict/api/v1/get-prediction/',methods=['POST'])
def return_predictions():
	if request.headers['Content-Type'] == 'text/plain':
		return "Text Message: " 
	elif request.headers['Content-Type'] == 'application/json':
		data = request.json
		result = json_normalize(data['hits']['hits'])
		return jsonify(predict.get_reccomended_ids(result))

	else:
		return "415 Unsupported Media Type ;)"
