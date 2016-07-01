#File paths
FILE_DATA = "./data/prod.csv"	#File to get the training data from
FILE_FEATURES = "./data/features.csv"	#File to get the feature list from
FILE_CLUSTER_MODEL = "./models/cluster_model.pkl"	#Name of the cluster model pickle file
FILE_SCALE_MODEL = "./models/scale_model.pkl"	#Name of the scale model pickle file
FILE_INDEX = "./data/index.csv"		#Name of the index file
FILE_PARTIAL_DATA = "./data/partialdata.csv"#File to perform partial fitting
FILE_CLUSTER_LOG = "logs/cluster.log" #Cluster log file
FILE_INDEX_DB = "db/index.db" 		#Index DB location
FILE_INPUT_DATA = "./data/input_data2.csv" 	#Sample input data for debugging purposes

#Index DB stuff
TABLE_NAME = "ids"
COL_INDEX = "indx"

#Preprocessing
SOURCE_TYPE = "csv"
SEPARATOR = ","
SCALER = 'standard'		#Change this to change the type of scaler

#Scaling types
SCALER_TYPE_STANDARD = 'standard'
SCALER_TYPE_MIN_MAX = 'min-max'
SCALER_TYPE_ROBUST = 'robust'

#Clustering constants
N_CLUSTERS=-1
CLUSTER_FACTOR=5
INIT='k-means++'
BATCH_SIZE=100 
N_INIT=10 
MAX_NO_OF_IMPROVEMENT=10
VERBOSE=0
RANDOM_STATE=0

#Logging
LOGGER_TOP = 'recommend'
LOGGER_CLUSTER = 'recommend.cluster'
LOGGER_PREPROCESS = 'recommend.preprocess'
LOGGER_PARTIAL_FIT = 'recommend.partial_fit'



