import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score

def show_PCA_graph(dataset,type="3d"):
	"""
	Does PCA on the dataset and plots pca1 vs pca2
	"""
	pca = PCA(n_components=2)
	transform = pca.fit_transform(dataset)
	#print pca.explained_variance_ratio_
	if type=="3d":	
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(transform[:,0],transform[:,1],range(transform.shape[0]))
	else:
		plt.scatter(transform[:,0],transform[:,1])

	plt.show()
	return

def elbow_test(c_start,c_end,step):
	"""
	Plots graph of inertia vs no of clusters
	"""
	inertia = []
	for i in xrange(c_start,c_end,step):
		model = cluster.KMeans(n_clusters=i)
		model.fit(train)
		inertia.append(model.inertia_)
	plt.plot(range(c_start,c_end,step),inertia)
	plt.xlabel("Number of Clusters")
	plt.ylabel("Inertia")
	plt.show()
	return

def silhouette_analysis(dataset):
	"""
	Plots the sillhoutte coefficient for a given dataset
	"""
	for n_clusters in xrange(2,18,2):
		X = dataset
		fig, ax1 = plt.subplots(1)
		ax1.set_xlim([-1, 1])
		ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(X)
		silhouette_avg = silhouette_score(X, cluster_labels)
		print "For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg
		sample_silhouette_values = silhouette_samples(X, cluster_labels)

		y_lower = 10
		for i in range(n_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
			    sample_silhouette_values[cluster_labels == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.spectral(float(i) / n_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper),
			                  0, ith_cluster_silhouette_values,
			                  facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhoutte score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		plt.savefig('../silhouette_analysis/cluster'+str(n_clusters)+'.png', bbox_inches='tight')


DATA_FILE_NAME = "data.csv"

columns = ['index'	,'type'	,'pid'	,'score'	,'id'	,'name'	,'date_entered'	,'date_modified'	,'description'	,'assigned_user_name'	,'deleted'	,'additional_rooms'	,'amenities_status'	,'available_from_date'	,'backend_status'	,'covered_area'	,'currency_type'	,'documents_status'	,'lease_duration_expected'	,'locality'	,'lat'	,'lon'	,'lock_in_period'	,'maintenance_charges'	,'no_of_balconies'	,'no_of_bathrooms'	,'owner_verified'	,'plot_area'	,'posted_on'	,'property_type'	,'property_verified'	,'rejection_comments'	,'rent_expected'	,'security_deposit'	,'state'	,'status'	,'suitable_time_to_call'	,'type_of_accomodation'	,'unit_of_measure'	,'year_of_construction1'	,'city'	,'content_team_status'	,'availabllity'	,'total_floors'	,'property_facing'	,'property_title'	,'unit_of_measure1'	,'floor'	,'lease_type_expected'	,'lease_subtype_expected1'	,'remarks'	,'servant_room'	,'servant_room_with_toilet'	,'flooring_type'	,'pin_code'	,'suitable_day'	,'periodicity_maintenance_bill'	,'proof_of_ownership_of_rental_p']
selected_columns = ['pid', 'name', 'covered_area', 'lease_duration_expected', 'lat', 'lon', 'lock_in_period', 'maintenance_charges', 'no_of_balconies', 'no_of_bathrooms', 'plot_area', 'rent_expected', 'security_deposit', 'type_of_accomodation', 'total_floors', 'floor', 'servant_room', 'servant_room_with_toilet']
#reqd_cols = ['pid', 'name', 'covered_area', 'lease_duration_expected', 'lat', 'lon', 'lock_in_period', 'maintenance_charges', 'plot_area', 'rent_expected', 'security_deposit', 'type_of_accomodation']
df = pd.read_csv(DATA_FILE_NAME,sep=',',header=0,names=selected_columns, usecols=selected_columns)
n_samples = df.pid.shape[0]
n_features = len(selected_columns)

#print "Samples= ",n_samples
#print "Features= ",n_features

dataset = np.zeros((n_samples,n_features-3))
for row in df.itertuples():
	for x in xrange(0,n_features-3):
		dataset[row[0],x] = row[x+3]

train = dataset.copy()
train = train[10:]
test = dataset.copy()
test = test[:10]

model = KMeans(n_clusters=10)

#TODO: model persistance (pickle)

labels = model.labels_
#print len(labels),dataset.shape,train.shape,test.shape
clusters = model.predict(test)
for j in xrange(len(test)):
	similar = []
	for i in xrange(len(labels)):
		if labels[i] == clusters[j]:
			similar.append(i+10)
	outf = df.ix[similar]
	outf.to_csv('output'+str(j)+'.csv',columns=selected_columns)
