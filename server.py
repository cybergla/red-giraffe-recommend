from flask import Flask, request, jsonify
from pandas.io.json import json_normalize
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
from logging.handlers import RotatingFileHandler

import predict
import config.constants as constants
import partialfit

app = Flask(__name__)

@app.route('/test')
def return_success():
	return "Success!"

@app.route('/predict/api/v1/get-prediction/',methods=['POST'])
def return_predictions():
	if request.headers['Content-Type'] == 'application/json':
		data = request.json
		p_id = data['hits']['hits'][0]['_id']			#Get requested property id
		result = json_normalize(data['hits']['hits'])	#Flatten JSON
		recs = predict.get_reccomended_ids(result)		#Get recommendations and remove requested property from it
		try:
			recs.remove(p_id)
		except Exception, e:
			app.logger.warning("Could not remove id from recs")
		response = jsonify(recs)						#Convert list to JSON
		return response, '200', {"Content-Type": "application/json"}

	else:
		app.logger.error("415: Unsupported Media Type. Expected JSON object")
		return "Unsupported Media Type. Expected JSON object",'415'

@app.route('/cluster/api/v1/partial-fit/',methods=['POST'])
def do_partial_fit():
	filename = request.form['filename']					#Get filename
	partialfit.fit('./data/'+filename)					#Do partital fit
	
	return "Done",'200'

#Error handlers
@app.errorhandler(500)
def internal_error(exception):
	app.logger.error(exception)
	return "Internal Server Error", 500

@app.errorhandler(400)
def internal_error(exception):
	app.logger.error(exception)
	return "Bad Request", 400

if __name__ == '__main__':
	if app.debug is not True:
		handler = RotatingFileHandler(constants.FILE_SERVER_LOG, maxBytes=1024*1024*100, backupCount=1)
		handler.setLevel(logging.INFO)
		handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
		app.logger.addHandler(handler)
	app.run()
