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

## Running the Predictions module
The Flask app can be run on a server for development purposes on `localhost:5000`
```
export FLASK_DEBUG=1                #Optional
python server.py
```

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
usage: main.py [-h] [--input-file INPUT_FILE]
               [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--mode {FULL,PARTIAL}]

Perform K Means clustering on a given dataset.

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE, -i INPUT_FILE
                        the file name of the input dataset
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: WARNING)
  --mode {FULL,PARTIAL}, -m {FULL,PARTIAL}
                        Type of clustering (default: FULL)
```
#### 1.1 Full
Clustering is done on the entire dataset and a new model + index is created. Should be used for large data

#### 1.2 Partial
Partial fit is done when there is incremental data to be fed into the model. Model is appended with the new labels, but the old data will **not** have its labels updated. Therefore, it should only be used for small amounts of data (< 10% of the size of the original dataset). Otherwise predictions might not be that accurate.

### 2. Predictions
Predicts the similar properties for a given property whose attributes are specified in the form json data.

Returns a list of property ids.

### 3. Server
A flask service which uses the predict module to give out recommendations.
The service is deployed as web service by nginx and gunicorn.

### 4 Config
#### 4.1 Constants
Keeps project level constants such as default filenames, field names etc.

### 5 Utils
#### 5.1 Preprocess
Contains all the preprocessing logic such as sanitising input, removing unnecessary columns, regex substitution on specified columns.

* To drop rows containing invalid values, specify that value in the `invalid_value` column in the features.csv table.
* All rows containing NA/NULL values will be dropped automatically
* To filter columns, specify the regular expression in the `regex` column and the substitute value in `substitution`

#### 5.2 Json_convert
Utility for conversion of input JSON data to program specific CSV format

#### 5.3 Read Data
Module to read data from multiple sources

### 6. Test
The test module is a small script that performs clustering and predictions using test data and outputs the results to the `results/` folder.
```
usage: test.py [-h] [--train-file TRAIN_FILE] [--test_file TEST_FILE] [-N N]

Test the workflow. Perform K Means clustering, then run predictions for N rows
of the dataset. Predictions are stored in results/output(x).csv

optional arguments:
  -h, --help            show this help message and exit
  --train-file TRAIN_FILE, -i TRAIN_FILE
                        the file name of the training dataset
  --test_file TEST_FILE, -t TEST_FILE
                        the file name of the testing dataset
  -N N                  number of testing samples to select from the testing
                        dataset (default: 5)
```

### 7. Data
#### 7.1 Features
Specify which features to choose from the data set for clustering. Format :-
```
+---------------+---------------+---------------+--------------+--------------+--------------+
| column_name   | invalid_value | regex         | substitution | regex        | substitution | ....
+---------------+---------------+---------------+--------------+--------------+--------------+
```