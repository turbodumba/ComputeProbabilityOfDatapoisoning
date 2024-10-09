from main import begin

testdatasets = ['Housing.csv', 'lungcancerpatient.csv', 'seattle-weather.csv']
poisoned_testdatasets = [dataset.replace('.csv', '_Poisoned.csv') for dataset in testdatasets]
output_name = "test_output.txt"
housingCategoryColumns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
                          'furnishingstatus']
housingNumericalColumns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
lungcancerCategoryColumns = ['Level', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards',
                             'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
                             'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                             'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
                             'Frequent Cold', 'Dry Cough', 'Snoring']
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
    begin(testdatasets, testdataset_dict, poisoned_testdatasets, output_name, iterations=2, percentage_at_start=0.05,
          increment=0.05)
