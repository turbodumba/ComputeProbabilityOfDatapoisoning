from main import poisonDatasets, getScores, printScoresAndDifferences

testdatasets = ['Housing.csv', 'lungcancerpatient.csv', 'seattle-weather.csv']
poisoned_testdatasets = [dataset.replace('.csv', '_Poisoned.csv') for dataset in testdatasets]
housingCategoryColumns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
housingNumericalColumns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
lungcancerCategoryColumns = ['Level', 'Gender', 'Air Pollution', 'Alcohol use','Dust Allergy', 'OccuPational Hazards', 'Genetic Risk','chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking','Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue','Weight Loss', 'Shortness of Breath', 'Wheezing','Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold','Dry Cough', 'Snoring']
lungcancerNumericalColumns = ['Age']
weatherCategoryColumns = ['weather']
weatherNumericalColumns = ['precipitation', 'temp_max', 'temp_min', 'wind']

c = 'category'
n = 'numerical'
t = 'targetCategorical'

testdataset_dict = {
    testdatasets[0]: {
        c: housingCategoryColumns,
        n: housingNumericalColumns,
        t: False
    },
    testdatasets[1]: {
        c: lungcancerCategoryColumns,
        n: lungcancerNumericalColumns,
        t: True
    },
    testdatasets[2]: {
        c: weatherCategoryColumns,
        n: weatherNumericalColumns,
        t: True
    }
}

if __name__ == '__main__':
    iterations = 20
    percentage_at_start = 0.05
    noise_level = 0.05
    increment = 0.05
    print(" Starting the algorithm with iterations: ", iterations, " manipulation_percentage: ", percentage_at_start,
          " noise_level: ", noise_level, " and increment: ", increment)
    originalList = getScores(testdatasets, testdataset_dict)
    print("=== The original scores finished calculating ===")
    poisonedList = []
    for i in range(iterations):
        poisonDatasets(testdatasets, percentage_at_start + i * increment, noise_level, testdataset_dict)
        poisonedList.append(getScores(poisoned_testdatasets, testdataset_dict))
        print("=== The ", i, " iteration score finished calculating ===")
    printScoresAndDifferences(testdatasets, originalList, poisonedList, percentage_at_start, increment)
    print("==== DONE ====")
