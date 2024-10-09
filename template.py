from main import begin, output_name

template_datasets = ['']
template_poisoned_datasets = [dataset.replace('.csv', '_Poisoned.csv') for dataset in template_datasets]
categoryColumns = ['']
numericalColumns = ['']
outputFileName = "templateFileName"

c = 'category'
n = 'numerical'
t = 'targetCategorical'

template_dataset_dict = {
    template_datasets[0]: {
        c: categoryColumns,
        n: numericalColumns,
        t: True
    },
}

if __name__ == '__main__':
    begin(template_datasets, template_dataset_dict, template_poisoned_datasets, outputFileName)
