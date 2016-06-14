import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import argparse
import utils.preprocess as preprocess
import config.constants as constants

parser = argparse.ArgumentParser(description='Perform K Means clustering on a given dataset.')
parser.add_argument('--input-file', '-i'c, type=str, help='the file name of the input dataset', default=constants.FILE_DATA)


args = parser.parse_args()

df = preprocess.get_data(constants.FILE_PARTIALDATA,1,to_append=True)

#Load scale model from disk
std_scale = joblib.load(constants.FILE_SCALE_MODEL)
dataset = std_scale.transform(df)

#Load cluster model from disk
model = joblib.load(constants.FILE_CLUSTER_MODEL)
lists=model.labels_
#Partial fit on the old trained model
model.partial_fit(dataset)
#Append the labels
lists=np.append(lists,model.labels_)
model.labels_=lists
joblib.dump(model, constants.FILE_CLUSTER_MODEL)

	

	
