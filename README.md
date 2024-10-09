# Computing the Probability of Data-poisoning in the dataset.
This is an algorithm, which was designed to compute a score, which is representative of the probability of poisoned data
being a part of the dataset.
It was developed as part of a Bachelor's thesis which is called "Computing the Trustworthiness Level of AI-based Models
Trained with Continuous Data".

## Prerequisites
The Python version which was used to develop this code is the version: 3.12.6

The following packages and versions were used during the development of the algorithm and need to be installed in order
to make the algorithm work correctly:
- Numpy: Version 1.26.4 (https://pypi.org/project/numpy/1.26.4/)
- Pandas: Version 2.2.2 (https://pypi.org/project/pandas/2.2.2/)
- Scikit-Learn: Version 1.5.1 (https://pypi.org/project/scikit-learn/1.5.1/)
- KModes: Version 0.12.2 (https://pypi.org/project/kmodes/0.12.2)

## Installation
There is no installation required, apart from being able to run Python code and being able to clone this repository.

## Usage
To use the algorithm, one can use the test.py class as an example of how to run the algorithm on whatever datasets the
user wants to run it on.
To use it for new datasets, the following steps can be taken:

1. Move the datasets, which should be tested on into the Datasets folder of the project.
2. Copy the template.py file in order to have a new class to fill in.
3. Fill in the datasets list with the names of the datasets.
4. Fill in the categorical columns with the names of the columns as seen in the csv file.
5. Fill in the numerical columns with the names of the columns as seen in the csv file.
6. Set an output file name
7. Now the last thing that can be adjusted is the 'begin' method, which can be filled with different parameters.

Possible parameters:

- iterations: Which is set to 20 by default. It represents how many iterations of poisoning and scoring the algorithm
  should do. (Integer)
- percentage_at_start: Which is set to 0.05 by default. It represents which manipulation percentage the algorithm starts
  with. Should be set between 0 & 1. (Float)
- increment: Which is set to 0.05 by default. It represents the increment of the manipulation percentage over the
  iterations. (Float)
  The poisoning can only be done between the value of 0 and 1, everything else will throw an error.
- noise_level: Which is set to 0.05 by default. It represents the noise level which is used in the calculation of the
  noise poisoned data. (Float)
- z_threshold: Which sets the threshold used by the Z-Score method to detect an outlier. Default value 3. (Float)
- binary_threshold: Which sets the threshold used for binary categories in the frequency outlier detection. Default is:
  0.3 (Float)
- category_threshold_multiple: Which sets the value, which the threshold will be multiplied by in the frequency outlier
  detection. Default is 0.5 (Float)
- high_cardinality_threshold: This value sets the threshold that is used in the frequency based outlier detection for
  high cardinality cases. Default is 0.01 (Float)
- categorical_unique_value_threshold: This represents the number of categories, which change the threshold from low to
  high cardinality. Default is 10 (Integer)
- rF_n_estimators: This represents the estimators used in the random forest algorithm. Default is 150 (Integer)
- dbscan_eps: This represents the epochs set in the DBSCAN algorithm. Default value is 0.5 (Float)
- dbscan_min_samples: This represents the minimum amount of samples needed for the DBSCAN algorithm. Default value is
  5 (Integer)
- ensemble_contamination: This represents the contamination percentage used in the ensemble method, to detect outliers.
  Default value is 0.1 (Float)
- kMeans_n_clusters: This represents the number of clusters set for the KMeans algorithm to start with. Default value is
  3 (Integer)
- kMeans_threshold: This represents the threshold set for the clusters in the KMeans algorithm to be classified as
  outliers. Default value is 0.05 (Float)

## Testing:
np.random.seed(42) was used in the data poisoning class, to ensure the poisoning was done in the same way for the
different tests.
To create the graphs with the CreateGraph class, one has to fill in the Combined_Scores.txt file with the results that shoul be plotted.
Copy and Paste from "=== Original Scores ===" to ==== DONE ====

