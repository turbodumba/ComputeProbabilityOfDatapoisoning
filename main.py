from DataPoisoning.DataPoisoning import createPoisonedDatasets, deletePoisonedData
from TestScores.Statistical_measures import calculateStatistics
import os

os.environ['OMP_NUM_THREADS'] = '5'

datasets = ['cardiovascular_risk.csv', 'obesityDataSet.csv', 'salary_2500.csv']
poisoned_datasets = [dataset.replace('.csv', '_Poisoned.csv') for dataset in datasets]
# cardioVascularCategoryColumns = ['TenYearCHD', 'education', 'sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
cardioVascularCategoryColumns = ['TenYearCHD']
cardioVascularNumericalColumns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI',
                                  'heartRate', 'glucose']
# obesityDataSetCategoryColumns = ['NObeyesdad', 'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
obesityDataSetCategoryColumns = ['NObeyesdad']
obesityDataSetNumericalColumns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
# salaryDatasetCategoryColumns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']
salaryDatasetCategoryColumns = ['salary']
salaryDatasetNumericalColumns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                 'hours-per-week']

c = 'category'
n = 'numerical'

datasets_dict = {
    datasets[0]: {
        c: cardioVascularCategoryColumns,
        n: cardioVascularNumericalColumns
    },
    datasets[1]: {
        c: obesityDataSetCategoryColumns,
        n: obesityDataSetNumericalColumns
    },
    datasets[2]: {
        c: salaryDatasetCategoryColumns,
        n: salaryDatasetNumericalColumns
    }
}


def poisonDatasets(datasetsList, percentage, noiselvl, datasetsdict):
    for dataset in datasetsList:
        createPoisonedDatasets(dataset, datasetsdict[dataset][c], datasetsdict[dataset][n],
                               percentage, noiselvl)


def getScores(filepaths, datasetsdict):
    scoreList = []
    keys = datasetsdict.keys()
    for key, filepath in zip(keys, filepaths):
        score = calculateStatistics(filepath, datasetsdict[key][c], datasetsdict[key][n])
        scoreList.append(score)
    return scoreList


def printScoresAndDifferences(datasetList, originalsScores, poisonedScores, startPerc, increments):
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


if __name__ == '__main__':
    iterations = 20
    percentage_at_start = 0.05
    noise_level = 0.05
    increment = 0.05
    print(" Starting the algorithm with iterations: ", iterations, " manipulation_percentage: ", percentage_at_start,
          " noise_level: ", noise_level, " and increment: ", increment)

    originalList = getScores(datasets, datasets_dict)
    print("=== The original scores finished calculating ===")
    poisonedList = []
    for i in range(iterations):
        poisonDatasets(datasets, percentage_at_start + i * increment, noise_level, datasets_dict)
        poisonedList.append(getScores(poisoned_datasets, datasets_dict))
        print("=== The ", i, " iteration score finished calculating ===")
    printScoresAndDifferences(datasets, originalList, poisonedList, percentage_at_start, increment)
    print("==== DONE ====")
