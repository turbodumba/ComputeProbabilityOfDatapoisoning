import os
import sys

from DataPoisoning.DataPoisoning import createPoisonedDatasets
from TestScores.Statistical_measures import calculateStatistics

os.environ['OMP_NUM_THREADS'] = '5'

main_datasets = ['cardiovascular_risk.csv', 'obesityDataSet.csv', 'salary_2500.csv']
main_poisoned_datasets = [dataset.replace('.csv', '_Poisoned.csv') for dataset in main_datasets]
output_name = "main_output.txt"

cardioVascularCategoryColumns = ['TenYearCHD', 'education', 'sex', 'is_smoking', 'BPMeds', 'prevalentStroke',
                                 'prevalentHyp', 'diabetes']
cardioVascularNumericalColumns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI',
                                  'heartRate', 'glucose']
obesityDataSetCategoryColumns = ['NObeyesdad', 'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
                                 'family_history_with_overweight', 'CAEC', 'MTRANS']
obesityDataSetNumericalColumns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
salaryDatasetCategoryColumns = ['salary', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'native-country']
salaryDatasetNumericalColumns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                 'hours-per-week']
c = 'category'
n = 'numerical'
t = 'targetCategorical'

main_datasets_dict = {
    main_datasets[0]: {
        c: cardioVascularCategoryColumns,
        n: cardioVascularNumericalColumns,
        t: True
    },
    main_datasets[1]: {
        c: obesityDataSetCategoryColumns,
        n: obesityDataSetNumericalColumns,
        t: True
    },
    main_datasets[2]: {
        c: salaryDatasetCategoryColumns,
        n: salaryDatasetNumericalColumns,
        t: True
    }
}


# The helper function which initiates the poisoning of the dataset.
def poisonDatasets(datasetsList, percentage, noiselvl, datasets_dict):
    for dataset in datasetsList:
        target = []
        if datasets_dict[dataset][t]:
            target = [datasets_dict[dataset][c][0]]
        createPoisonedDatasets(dataset, target, datasets_dict[dataset][c], datasets_dict[dataset][n],
                               percentage, noiselvl)


# The helper function which initiates the scoring of the original and poisoned datasets.
def getScores(filepaths, datasets_dict, z_threshold, binary_threshold,
              category_threshold_multiple,
              high_cardinality_threshold, categorical_unique_value_threshold, rF_n_estimators,
              dbscan_eps, dbscan_min_samples, ensemble_contamination, kMeans_n_clusters, kMeans_threshold):
    scoreList = []
    keys = datasets_dict.keys()
    for key, filepath in zip(keys, filepaths):
        score = calculateStatistics(filepath, datasets_dict[key][c], datasets_dict[key][n], z_threshold,
                                    binary_threshold, category_threshold_multiple, high_cardinality_threshold,
                                    categorical_unique_value_threshold, rF_n_estimators, dbscan_eps, dbscan_min_samples,
                                    ensemble_contamination, kMeans_n_clusters, kMeans_threshold)
        scoreList.append(score)
    return scoreList


# Helper function, which is used to print the scores of all the datasets one by one.
def printScores(datasetList, originalsScores, poisonedScores, startPerc, increments):
    print("=== Original Scores ===")
    for k, score in enumerate(originalsScores):
        print(datasetList[k], " got this score: ", score)
    perc = startPerc
    print("=== Poisoned Scores ===")
    for poisoned in poisonedScores:
        print("Poisoned Datasets with manipulating percentage: ", round(perc, 2))
        for k, score in enumerate(poisoned):
            print(datasetList[k], " got this poisoned scores: ", score)
        perc = perc + increments


def printScoresAndDifferences(datasetList, originalsScores, poisonedScores, startPerc, increments, filename):
    # Print to the console
    printScores(datasetList, originalsScores, poisonedScores, startPerc, increments)
    # Get the correct file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the full path to the file
    file_path = os.path.join(script_dir, filename)
    # Open the file for writing
    with open(file_path, 'w') as f:
        sys.stdout = f  # Write to the file
        printScores(datasetList, originalsScores, poisonedScores, startPerc, increments)
        print("==== DONE ====")
        sys.stdout = sys.__stdout__  # Restore stdout to the console


# The helper function that initiates all the steps, from poisoning, to scoring and printing the results.
def begin(datasets_used, datasets_dict_used, poisoned_datasets_used, outputfile_name, iterations=20,
          percentage_at_start=0.05, increment=0.05, noise_level=0.05,
          z_threshold=3, binary_threshold=0.3,
          category_threshold_multiple=0.5,
          high_cardinality_threshold=0.01, categorical_unique_value_threshold=10, rF_n_estimators=150,
          dbscan_eps=0.5, dbscan_min_samples=5, ensemble_contamination=0.1, kMeans_n_clusters=3, kMeans_threshold=0.05):
    print(" Starting the algorithm with iterations: ", iterations, " manipulation_percentage: ", percentage_at_start,
          " noise_level: ", noise_level, " and increment: ", increment)

    originalList = getScores(datasets_used, datasets_dict_used, z_threshold, binary_threshold,
                             category_threshold_multiple, high_cardinality_threshold,
                             categorical_unique_value_threshold, rF_n_estimators, dbscan_eps, dbscan_min_samples,
                             ensemble_contamination, kMeans_n_clusters, kMeans_threshold)
    print("=== The original scores finished calculating ===")
    poisonedList = []
    for i in range(iterations):
        poisonDatasets(datasets_used, percentage_at_start + i * increment, noise_level, datasets_dict_used)
        poisonedList.append(getScores(poisoned_datasets_used, datasets_dict_used, z_threshold, binary_threshold,
                                      category_threshold_multiple, high_cardinality_threshold,
                                      categorical_unique_value_threshold, rF_n_estimators, dbscan_eps,
                                      dbscan_min_samples, ensemble_contamination, kMeans_n_clusters, kMeans_threshold))
        print("=== The ", i, " iteration score finished calculating ===")
    printScoresAndDifferences(datasets_used, originalList, poisonedList, percentage_at_start, increment,
                              outputfile_name)


if __name__ == '__main__':
    begin(main_datasets, main_datasets_dict, main_poisoned_datasets, output_name)
