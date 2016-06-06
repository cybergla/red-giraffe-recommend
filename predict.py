import numpy as np
import pandas as pd
import re
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

FILE_DATA = "input_data.csv"	#File to get the test data from
FILE_FEATURES = "features.csv"	#File to get the feature list from

df_features = pd.read_csv(FILE_FEATURES,sep=',',header=None)
selected_columns = df_features[0].values

df = pd.read_csv(FILE_DATA,sep=',',header=0,usecols=selected_columns)

#Data preprocessing
for index, row in df.iterrows():
	df.loc[index,"_source/type_of_accomodation"] = df.loc[index,"_source/type_of_accomodation"].replace("bhk","").replace("rk","")
	df.loc[index,"_source/floor"] = re.sub(r"\D*","",df.loc[index,"_source/floor"])

#Load scale model from disk
std_scale = joblib.load('scale_model.pkl')
#Normalize data
test = std_scale.transform(df[selected_columns[2:]])
#Load cluster model from disk
model = joblib.load('cluster_model.pkl')
#Get labels of cluster model
labels = model.labels_
#Predict cluster no. of test data
clusters = model.predict(test)
df_in = pd.read_csv("result.csv",sep=',',header=0)

for j in xrange(len(test)):
		similar = []
		for i in xrange(len(labels)):
		       if labels[i] == clusters[j]:
		               similar.append(i)
		outf = df_in.ix[similar]
		outf.to_csv('./results/output'+str(j)+'.csv')