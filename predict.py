import numpy as np
import pandas as pd
import re
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

FILE_DATA = "input_data.csv"
FILE_FEATURES = "features.csv"

df_features = pd.read_csv(FILE_FEATURES,sep=',',header=None)
selected_columns = df_features[0].values

df = pd.read_csv(FILE_DATA,sep=',',header=0,usecols=selected_columns)
n_samples = df._id.shape[0]
n_features = len(selected_columns)

for index, row in df.iterrows():
	df.loc[index,"_source/type_of_accomodation"] = df.loc[index,"_source/type_of_accomodation"].replace("bhk","").replace("rk","")
	df.loc[index,"_source/floor"] = re.sub(r"\D*","",df.loc[index,"_source/floor"])

std_scale = joblib.load('scale_model.pkl')
test = std_scale.transform(df[selected_columns[2:]])
model = joblib.load('cluster_model.pkl')
labels = model.labels_
#print labels, labels.shape
#print len(labels),dataset.shape,dataset.shape,test.shape
clusters = model.predict(test)
print clusters

df_in = pd.read_csv("data.csv",sep=',',header=0)

for j in xrange(len(test)):
		similar = []
		for i in xrange(len(labels)):
		       if labels[i] == clusters[j]:
		               similar.append(i)
		outf = df_in.ix[similar]
		outf.to_csv('./results/output'+str(j)+'.csv')