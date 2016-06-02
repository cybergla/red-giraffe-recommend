import numpy as np
import pandas as pd
from sklearn import cluster
from matplotlib import pyplot

names = ['pid', 'name', 'covered_area', 'lease_duration_expected', 'lat', 'lon', 'lock_in_period', 'maintenance_charges', 'no_of_balconies', 'no_of_barooms', 'plot_area', 'rent_expected', 'security_deposit', 'type_of_accomodation', 'total_floors', 'floor', 'servant_room', 'servant_room_wi_toilet']
df = pd.read_csv('data.csv',sep=',',header=1,names=names)

n_samples = df.pid.unique().shape[0]
n_features = len(names)

#print "Samples= ",n_samples
#print "Features= ",n_features

dataset = np.zeros((n_samples,n_features-3))
for row in df.itertuples():
	for x in xrange(0,n_features-3):
		dataset[row[0],x] = row[x+3]

k_means = cluster.KMeans(n_clusters=8)
k_means.fit(dataset)
labels = k_means.labels_
centroids = k_means.cluster_centers_

print labels
print centroids
