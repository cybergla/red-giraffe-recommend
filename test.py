import os
import pandas as pd

import predict
import utils.preprocess as preprocess
import config.constants as constants

FILE_DATA = "./data/prod.csv"

#os.system("python cluster.py -i "+FILE_DATA)

selected_columns = preprocess.get_features()
df = pd.read_csv(FILE_DATA,usecols=selected_columns)
df = df.ix[0]
print df, type(df)
#for i in xrange(df.shape[0]):
predict.get_reccomended_ids(df,FILE_DATA).to_csv("./results/output "+str(i)+".csv")