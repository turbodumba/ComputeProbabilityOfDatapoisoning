import os

import numpy as np
import pandas as pd


def fill_categorical_with_random_sample(dataset, columns_to_fill):
    for column in columns_to_fill:
        non_null_values = dataset[column].dropna().values
        random_samples = np.random.choice(non_null_values, size=len(dataset))
        dataset[column] = dataset[column].fillna(pd.Series(random_samples))
    return dataset


def poison_labels(dataset, columns_to_poison, flip_percentage=0.1):
    for label_column in columns_to_poison:
        valid_indices = dataset[~dataset[label_column].isna()].index
        num_poisoned = int(flip_percentage * len(valid_indices))
        flip_indices = np.random.choice(valid_indices, size=num_poisoned, replace=False)
        possible_labels = dataset[label_column].dropna().unique().tolist()

        for idx in flip_indices:
            current_label = dataset.loc[idx, label_column]
            new_label = np.random.choice([label for label in possible_labels if label != current_label])
            dataset.loc[idx, label_column] = new_label

    return dataset


def poison_with_random_noise(dataset, columns_to_poison, noise_percentage=0.1, noise_level=0.05):
    num_poisoned = int(noise_percentage * len(dataset))
    noise_indices = np.random.choice(dataset.index, size=num_poisoned, replace=False)

    for column in columns_to_poison:
        noise = np.random.normal(0, noise_level, size=num_poisoned)
        if pd.api.types.is_integer_dtype(dataset[column]):
            noise = noise.astype(int)
        elif pd.api.types.is_float_dtype(dataset[column]):
            noise = noise.astype(float)
        dataset.loc[noise_indices, column] += noise

    return dataset


def poison_with_out_of_distribution_data(dataset, columns_to_poison, categoricalColumns, ood_percentage=0.1):
    num_poisoned = int(ood_percentage * len(dataset))
    ood_data = {}
    for column in columns_to_poison:
        ood_data_point = np.random.uniform(dataset[column].min() * 1.5, dataset[column].max() * 1.5, num_poisoned)
        if pd.api.types.is_integer_dtype(dataset[column]):
            ood_data_point = ood_data_point.astype(int)
        elif pd.api.types.is_float_dtype(dataset[column]):
            ood_data_point = ood_data_point.astype(float)
        ood_data[column] = ood_data_point
    ood_df = pd.DataFrame(ood_data)
    poisoned_dataset = pd.concat([dataset, ood_df], ignore_index=True)

    # Fill missing categorical columns
    poisoned_dataset = fill_categorical_with_random_sample(poisoned_dataset, categoricalColumns)
    return poisoned_dataset


def poison_numerical_columns(dataset, columns_to_poison, manipulation_percentage, noise_level, categoricalColumns):
    poisoned_dataset = poison_with_random_noise(dataset, columns_to_poison, manipulation_percentage,
                                                noise_level)
    poisoned_dataset = poison_with_out_of_distribution_data(poisoned_dataset, columns_to_poison, categoricalColumns,
                                                            manipulation_percentage)
    return poisoned_dataset


def createPoisonedDatasets(filepath, targetColumns, categoricalColumns, numericalColumns, manipulation_percentage=0.1,
                           noise_level=0.01):
    np.random.seed(42)
    # Get the absolute path to the Datasets folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'Datasets')

    # Change directory to the dataset folder
    os.chdir(datasets_dir)

    # Read the CSV file
    dataset = pd.read_csv(filepath)
    dataset = poison_labels(dataset, targetColumns, manipulation_percentage)
    dataset = poison_numerical_columns(dataset, numericalColumns, manipulation_percentage, noise_level,
                                       categoricalColumns)
    dataset.to_csv(filepath.replace('.csv', '_Poisoned.csv'), index=False)
    return None


def deletePoisonedData(file_paths):
    # Get the absolute path to the Datasets folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'Datasets')
    # Change directory to the dataset folder
    os.chdir(datasets_dir)
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
