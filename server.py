from flask import Flask, request, jsonify
from pandas.io.json import json_normalize
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

if __name__ == '__main__':
	app.run()
