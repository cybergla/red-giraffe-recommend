RG Recommender System
=====================
This is the Property Recommender system developed for Red Girraffe. It works by performing K Means clustering on a large number of properties and returns a list of similar properties.

## Requirements
* Python 2.7
* pip
* Git

## Installation
1. Clone this repository using Git
2. Install `python, pip, python-dev` packages from the package manager on your machine
3. Install project dependencies using  `pip install -r requirements.txt`

## Project Structure
```
recommend/
     __init.py__
    config/
        __init.py__
        constants.py
    utils/
        __init.py__
        graph.py
        preprocess.py
        json_convert.py
    logs/
    data/
    results/
    models/
    cluster.py
    predict.py
    partialfit.py
    test.py
    requirements.txt
    README.md
```

## Usage
### 1. Clustering
Cluster using K-Means algorithm on the dataset given in `data/<file-name.csv>`
```
usage: cluster.py [-h] [--input-file INPUT_FILE] 
                  [--n-clusters N_CLUSTERS]
                  [--cluster-factor CLUSTER_FACTOR]
                  [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Perform K Means clustering on a given dataset.

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE, -i INPUT_FILE
                        the file name of the input dataset
  --n-clusters N_CLUSTERS, -N N_CLUSTERS
                        number of clusters (default: no. of samples/cf)
  --cluster-factor CLUSTER_FACTOR, -cf CLUSTER_FACTOR
                        determines number of clusters (default: 5)
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: WARNING)
```
### 2. Predictions
Predicts the similar properties for a given property whose attributes are specified in the form json data.

Returns a list of property ids.
### 3. Partial Fit
### 4. Server
A flask service which uses the predict module to give out recommendations.
The service is deployed as web service by nginx and gunicorn.
### 5 Config
#### 5.1 Constants
### 6 Utils
#### 6.1 Preprocess
#### 6.2 Json_convert
Utility for convertion of input json data to program specific csv format