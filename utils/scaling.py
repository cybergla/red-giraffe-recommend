from sklearn import preprocessing
from sklearn.externals import joblib
import config.constants as constants
import sys
sys.path.append('.')
sys.path.append('..')
def scale_data(df):
	std_scale = get_scaler().fit(df)
	#Save normalized model to disk
	joblib.dump(std_scale,constants.FILE_SCALE_MODEL)
	return std_scale.transform(df)
def scale_new_data(df):
	std_scale = joblib.load(constants.FILE_SCALE_MODEL)
	return std_scale.transform(df)
def get_scaler(scale=constants.SCALER):
	if(scale=='standard'):
		return preprocessing.StandardScaler()