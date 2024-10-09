from DataPoisoning.DataPoisoning import createPoisonedDatasets, deletePoisonedData
from TestScores.Statistical_measures import calculateStatistics
import os
import sys

os.environ['OMP_NUM_THREADS'] = '5'

main_datasets = ['cardiovascular_risk.csv', 'obesityDataSet.csv', 'salary_2500.csv']
main_poisoned_datasets = [dataset.replace('.csv', '_Poisoned.csv') for dataset in main_datasets]

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


def poisonDatasets(datasetsList, percentage, noiselvl, datasets_dict):
    for dataset in datasetsList:
        target = []
        if datasets_dict[dataset][t]:
            target = [datasets_dict[dataset][c][0]]
        createPoisonedDatasets(dataset, target, datasets_dict[dataset][c], datasets_dict[dataset][n],
                               percentage, noiselvl)


def getScores(filepaths, datasets_dict):
    scoreList = []
    keys = datasets_dict.keys()
    for key, filepath in zip(keys, filepaths):
        score = calculateStatistics(filepath, datasets_dict[key][c], datasets_dict[key][n])
        scoreList.append(score)
    return scoreList

#Helper function, which is used to print the scores of all the datasets one by one.
def printScores(datasetList, originalsScores, poisonedScores, startPerc, increments):
    print("=== Original Scores ===")
    for k, score in enumerate(originalsScores):
        print(datasetList[k], " got this score: ", score)
    perc = startPerc
    print("=== Poisoned Scores ===")
    for poisoned in poisonedScores:
        print("Poisoned Datasets with manipulating percentage: ", round(perc, 2))
        for k, score in enumerate(poisoned):
            print(datasetList[k], " got this difference: ", (score - originalsScores[k]))
        perc = perc + increments

def printScoresAndDifferences(datasetList, originalsScores, poisonedScores, startPerc, increments, filename):
    # Print to the console
    printScores(datasetList, originalsScores, poisonedScores, startPerc, increments)

    # Open the file for writing
    with open(filename, 'w') as f:
        sys.stdout = f  # Redirect stdout to the file
        printScores(datasetList, originalsScores, poisonedScores, startPerc, increments)
        sys.stdout = sys.__stdout__  # Restore stdout to the console


def begin(datasets_used, datasets_dict_used):
    print(" Starting the algorithm with iterations: ", iterations, " manipulation_percentage: ", percentage_at_start,
          " noise_level: ", noise_level, " and increment: ", increment)

    originalList = getScores(datasets_used, datasets_dict_used)
    print("=== The original scores finished calculating ===")
    poisonedList = []
    for i in range(iterations):
        poisonDatasets(datasets_used, percentage_at_start + i * increment, noise_level, datasets_dict_used)
        poisonedList.append(getScores(main_poisoned_datasets, datasets_dict_used))
        print("=== The ", i, " iteration score finished calculating ===")
    printScoresAndDifferences(datasets_used, originalList, poisonedList, percentage_at_start, increment,
                              "main_output.txt")
    print("==== DONE ====")


if __name__ == '__main__':
    iterations = 20
    percentage_at_start = 0.05
    noise_level = 0.05
    increment = 0.05
    begin(main_datasets, main_datasets_dict)
